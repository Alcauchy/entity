#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

//#include "utils/numerics.h"
#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

    // field initializer
    template <Dimension D>
    struct Field {
    Field (real_t kx, real_t ampl)
      : kx { kx }, amplitude { ampl } {}

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return amplitude * math::sin(kx  * x_Ph[0]); // function of x1 coordinate
    }
    // you may skip other field components if you don't need them

  private:
    const real_t kx, amplitude;
  };


  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temp_1, temp_2;
    const real_t drift_u_1, drift_u_2;
    const real_t kx, amplitude;
    Field<D> init_fields;
    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temp_1 { p.template get<real_t>("setup.temp_1") }
      , temp_2 { p.template get<real_t>("setup.temp_2") }
      , drift_u_1 { p.template get<real_t>("setup.drift_u_1") }
      , drift_u_2 { p.template get<real_t>("setup.drift_u_2") }
      , kx { p.template get<real_t>("setup.kx") }
      , amplitude { p.template get<real_t>("setup.amplitude") }
      , init_fields{kx,amplitude} 
      {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist_1 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp_1,
                                                        drift_u_1,
                                                        in::x1,
							false);
      
      const auto injector_1 = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist_1,
        { 1, 2 });
      arch::InjectUniform<S, M, decltype(injector_1)>(
        params,
        local_domain,
        injector_1,
        ONE);
    }
  };

} // namespace user

#endif
