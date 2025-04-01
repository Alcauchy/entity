#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

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

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temp_1 { p.template get<real_t>("setup.temp_1") }
      , temp_2 { p.template get<real_t>("setup.temp_2") }
      , drift_u_1 { p.template get<real_t>("setup.drift_u_1") }
      , drift_u_2 { p.template get<real_t>("setup.drift_u_2") } {}

    inline void InitPrtls(Domain<S, M>& domain) {

      int              rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // if (rank != 0) {
      //   return;
      // }

      auto& species_e = domain.species[0];

      array_t<std::size_t> elec_ind("elec_ind");

      auto offset_e = species_e.npart();

      auto ux1_e    = species_e.ux1;
      auto ux2_e    = species_e.ux2;
      auto ux3_e    = species_e.ux3;
      auto i1_e     = species_e.i1;
      auto i2_e     = species_e.i2;
      auto dx1_e    = species_e.dx1;
      auto dx2_e    = species_e.dx2;
      auto phi_e    = species_e.phi;
      auto weight_e = species_e.weight;
      auto tag_e    = species_e.tag;

      int nseed = 1;

      Kokkos::parallel_for("init_particles", nseed, KOKKOS_LAMBDA(const int& s) {

        // ToDo: fix this
        auto i1_ = math::floor(17);
        auto i2_ = math::floor(33);
        auto dx1_ = HALF;
        auto dx2_ = HALF;


            auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);

            i1_e(elec_p + offset_e) = i1_;
            dx1_e(elec_p + offset_e) = dx1_;
            i2_e(elec_p + offset_e) = i2_;
            dx2_e(elec_p + offset_e) = dx2_;
            ux1_e(elec_p + offset_e) = -0.3456;
            ux2_e(elec_p + offset_e) = 0.6678;
            ux3_e(elec_p + offset_e) = ZERO;
            weight_e(elec_p + offset_e) = ONE;
            tag_e(elec_p + offset_e) = ParticleTag::alive;


        });


          auto elec_ind_h = Kokkos::create_mirror(elec_ind);
          Kokkos::deep_copy(elec_ind_h, elec_ind);
          species_e.set_npart(offset_e + elec_ind_h());

    }
  };

} // namespace user

#endif
