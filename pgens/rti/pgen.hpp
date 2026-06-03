#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <utility>
#include <vector>

namespace user {
  using namespace ntt;

  // -----------------------------------------------------------------------
  // Magnetic field initializer (Follows Zhdankin+2023)
  //
  // Hot cavity  (y < y_c):  B_x = +B_in,  B_z = B_hz  (uniform)
  // Cold slab   (y > y_c):  B_x = -B_in,  B_z = B_cz  (uniform)
  // Rotation zone [rot_lo, rot_hi] = [y_c - L_y/8, y_c + L_y/8]:
  //   linearly interpolate components, then rescale to maintain magnitude
  // -----------------------------------------------------------------------
  template <Dimension D>
  struct InitFields {
    real_t B_in;   // in-plane field component (hot: +B_in, cold: -B_in)
    real_t B_hz;   // hot out-of-plane (z) component
    real_t B_cz;   // cold out-of-plane (z) component
    real_t B_h;    // hot field magnitude
    real_t B_c;    // cold field magnitude 
    real_t y_c;    // interface y-position
    real_t rot_lo; // lower edge of rotation zone = y_c - L_y/8
    real_t rot_hi; // upper edge of rotation zone = y_c + L_y/8

    InitFields(real_t B_in_,
               real_t B_hz_,
               real_t B_cz_,
               real_t B_h_,
               real_t B_c_,
               real_t y_c_,
               real_t rot_lo_,
               real_t rot_hi_)
      : B_in { B_in_ }
      , B_hz { B_hz_ }
      , B_cz { B_cz_ }
      , B_h { B_h_ }
      , B_c { B_c_ }
      , y_c { y_c_ }
      , rot_lo { rot_lo_ }
      , rot_hi { rot_hi_ } {}

    // B_x
    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      if (x_Ph[1] <= rot_lo) {
        return B_in;
      } else if (x_Ph[1] >= rot_hi) {
        return -B_in;
      } else {
        const auto t      = (x_Ph[1] - rot_lo) / (rot_hi - rot_lo); // needed for linear interpolation
        const auto Bx_i   = (ONE - t) * B_in - t * B_in; // Linear interpolation
        const auto Bz_i   = (ONE - t) * B_hz + t * B_cz; // Linear interpolation
        const auto B_mag  = math::sqrt(SQR(Bx_i) + SQR(Bz_i));
        const auto target = (x_Ph[1] < y_c) ? B_h : B_c; // Rescaling to a constant magnitude in cold or hot region
        return Bx_i * target / B_mag;
      }
    }
    // B_y is set to zero
    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    // B_z, same as B_x
    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      if (x_Ph[1] <= rot_lo) {
        return B_hz;
      } else if (x_Ph[1] >= rot_hi) {
        return B_cz;
      } else {
        const auto t      = (x_Ph[1] - rot_lo) / (rot_hi - rot_lo);
        const auto Bx_i   = (ONE - t) * B_in - t * B_in;
        const auto Bz_i   = (ONE - t) * B_hz + t * B_cz;
        const auto B_mag  = math::sqrt(SQR(Bx_i) + SQR(Bz_i));
        const auto target = (x_Ph[1] < y_c) ? B_h : B_c;
        return Bz_i * target / B_mag;
      }
    }
  };

  // -----------------------------------------------------------------------
  // Uniform gravity in -y with no-gravity buffer near y-boundaries.
  // -----------------------------------------------------------------------
  template <Dimension D>
  struct Gravity {
    real_t g;
    real_t y_min, y_max, ds_grav;
    const std::vector<spidx_t> species { 1, 2, 3, 4 };

    Gravity(real_t g_, real_t y_min_, real_t y_max_, real_t ds_grav_)
      : g { g_ }
      , y_min { y_min_ }
      , y_max { y_max_ }
      , ds_grav { ds_grav_ } {}

    Inline auto fx1(spidx_t, simtime_t, const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto fx2(spidx_t, simtime_t, const coord_t<D>& x_Ph) const -> real_t {
      if (x_Ph[1] < y_min + ds_grav or x_Ph[1] > y_max - ds_grav) {
        return ZERO;
      }
      return -g;
    }

    Inline auto fx3(spidx_t, simtime_t, const coord_t<D>&) const -> real_t {
      return ZERO;
    }
  };

  // -----------------------------------------------------------------------
  // Boundary field functor for MatchFieldsInX2. - THIS IS NOT USED RIGHT NOW FOR THE CONDUCTOR BOUNDARIES
  // Freezes the y-boundary ghost cells to the uniform hot/cold field values.
  // -----------------------------------------------------------------------
  template <Dimension D>
  struct BoundaryFieldsX2 {
    real_t B_in, B_hz, B_cz, y_c;

    BoundaryFieldsX2(real_t B_in_, real_t B_hz_, real_t B_cz_, real_t y_c_)
      : B_in { B_in_ }
      , B_hz { B_hz_ }
      , B_cz { B_cz_ }
      , y_c { y_c_ } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return (x_Ph[1] < y_c) ? B_in : -B_in;
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return (x_Ph[1] < y_c) ? B_hz : B_cz;
    }
  };

  // -----------------------------------------------------------------------
  // Problem generator
  // -----------------------------------------------------------------------
  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines {
      traits::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_2D, Dim::_3D>::value
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    // --- input parameters (from [setup]) ---
    const real_t nc_over_nh; // density ratio n_cold / n_hot
    const real_t theta_rot;  // magnetic shear angle [rad]
    const real_t T_int;      // temperature at the interface [mc²]

    // --- fiducial constants ---
    const real_t B_c    { ONE };
    const real_t n_cold { ONE };

    // --- derived from scales, see entity wiki on normalization ---
    const real_t sigma_0; // (skindepth0/larmor0)^2
    const real_t beta_c;  // 2 n_cold T_int / (B_c^2 sigma_0)

    // --- domain geometry (derived from metadomain) ---
    const real_t y_min, y_max, y_c, L_y;

    // --- derived field quantities ---
    const real_t B_h;   // hot-region field magnitude
    const real_t B_in;  // in-plane component (eq. 1 of Zhdankin+2023)
    const real_t B_hz;  // hot out-of-plane component
    const real_t B_cz;  // cold out-of-plane component

    // --- gravitational acceleration and buffer zone ---
    const real_t gravity;
    const real_t ds_grav; // gravity-free buffer near y-boundaries

    InitFields<D>  init_flds;  // field initializer
    Gravity<D>     ext_force;  // applies external force

    inline PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , nc_over_nh { p.template get<real_t>("setup.nc_over_nh",
                       static_cast<real_t>(511.0)) }
      , theta_rot  { p.template get<real_t>("setup.theta_rot",
                       static_cast<real_t>(constant::PI * 0.25)) }
      , T_int      { p.template get<real_t>("setup.T_int",
                       static_cast<real_t>(INV_16)) }
      , sigma_0    { SQR(p.template get<real_t>("scales.skindepth0") /
                         p.template get<real_t>("scales.larmor0")) }
      , beta_c     { TWO * n_cold * T_int / SQR(B_c) / sigma_0 }
      , y_min      { m.mesh().extent(in::x2).first }
      , y_max      { m.mesh().extent(in::x2).second }
      , y_c        { HALF * (m.mesh().extent(in::x2).first +
                             m.mesh().extent(in::x2).second) }
      , L_y        { m.mesh().extent(in::x2).second -
                     m.mesh().extent(in::x2).first }
      , B_h        { B_c * math::sqrt(ONE +
                       (ONE - ONE / nc_over_nh) * beta_c) }
      , B_in       { B_h * B_c * math::sin(theta_rot) /
                     math::sqrt(SQR(B_h) + SQR(B_c) +
                                TWO * B_h * B_c * math::cos(theta_rot)) }
      , B_hz       { math::sqrt(math::abs(SQR(B_h) - SQR(B_in))) }
      , B_cz       { math::sqrt(math::abs(SQR(B_c) - SQR(B_in))) }
      , gravity    { p.template get<real_t>("setup.gravity",
                       static_cast<real_t>(2.4) * T_int / L_y) }
      , ds_grav    { p.template get<real_t>("setup.ds_grav",
                       L_y * static_cast<real_t>(0.125)) }
      , init_flds  { B_in, B_hz, B_cz, B_h, B_c, y_c,
                     y_c - L_y * static_cast<real_t>(0.125),
                     y_c + L_y * static_cast<real_t>(0.125) }
      , ext_force  { gravity, y_min, y_max, ds_grav } {}

    auto MatchFieldsInX2(simtime_t) const -> BoundaryFieldsX2<D> {
      return BoundaryFieldsX2<D> { B_in, B_hz, B_cz, y_c };
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      // Bounding boxes for the two density regions (physical coords)
      boundaries_t<real_t> hot_box, cold_box;
      hot_box.push_back(Range::All); // x-boundaries of the injection box
      hot_box.push_back({ y_min, y_c }); // y-boundaries of the injection box
      cold_box.push_back(Range::All); // x-boundaries of the injection box
      cold_box.push_back({ y_c, y_max });
      if constexpr (D == Dim::_3D) {
        hot_box.push_back(Range::All); //z-boundaries if the simulation is in 3D 
        cold_box.push_back(Range::All);
      }

      const auto n_h = n_cold / nc_over_nh; // density of hot plasma
      const auto no_drift = std::make_pair(std::vector<real_t> { ZERO, ZERO, ZERO },
                                           std::vector<real_t> { ZERO, ZERO, ZERO }); //no drift passed to maxwell distributions

      // Hot region (y < y_c): species 1 (e-) + 2 (ions)
      arch::InjectUniformMaxwellians<S, M>(
        params, local_domain, n_h, { T_int, T_int }, { 1, 2 }, no_drift, false, hot_box);
      // Cold region (y > y_c): species 3 (e-) + 4 (ions)
      arch::InjectUniformMaxwellians<S, M>(
        params, local_domain, n_cold, { T_int, T_int }, { 3, 4 }, no_drift, false, cold_box);

      // Rescale velocities for hydrostatic temperature stratification:
      // T(y) = T_int - m * g * (y - y_c) in the gravity zone,
      // velocities remain unchanged in no-gravity buffer zones.
      const auto metric_    = local_domain.mesh.metric;
      const auto g_         = gravity;
      const auto T_         = T_int;
      const auto yc_        = y_c;
      const auto y_grav_lo_ = y_min + ds_grav;
      const auto y_grav_hi_ = y_max - ds_grav;

      for (spidx_t sp = 1; sp <= 4; ++sp) {
        auto&      prtl  = local_domain.species[sp - 1];
        const auto np    = prtl.npart();
        if (np == 0) {
          continue;
        }
        const auto mass_ = prtl.mass();
        auto       i2_   = prtl.i2;
        auto       dx2_  = prtl.dx2;
        auto       ux1_  = prtl.ux1;
        auto       ux2_  = prtl.ux2;
        auto       ux3_  = prtl.ux3;
        auto       tag_  = prtl.tag;

        Kokkos::parallel_for(
          "RTI_TempRescale", np, Lambda(npart_t p) {
            if (tag_(p) == ParticleTag::dead) {
              return;
            }
            coord_t<D> x_Cd { ZERO }, x_Ph { ZERO };
            x_Cd[1] = static_cast<real_t>(i2_(p)) +
                      static_cast<real_t>(dx2_(p));
            metric_.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);

            // checking if rescaling is needed
            const auto y_eff = x_Ph[1] < y_grav_lo_ ? y_grav_lo_
                             : x_Ph[1] > y_grav_hi_ ? y_grav_hi_
                             : x_Ph[1];
            const auto T_loc = T_ - mass_ * g_ * (y_eff - yc_);
            if (T_loc <= ZERO) {
              return;
            }
            const auto scale = math::sqrt(T_loc / T_);
            ux1_(p) *= scale;
            ux2_(p) *= scale;
            ux3_(p) *= scale;
          });
      }
    }
  };

} // namespace user

#endif
