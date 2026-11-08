// Port of the metadata application path of src/speaker_renderer.py HEAD.
//
// The metadata path consumes the full per-object per-block OAMD frame state and
// schedules block targets frame by frame: object gain/inactive, zone/elevation,
// snap, divergence, size kernel fusion, priority size budget, ISF basis rows and
// bed channel distribution, with the same float64 gain ramps and block alignment
// as the reference renderer.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "eac3joc_core.h"
#include "foundation/status.h"
#include "oamd/oamd_parser.h"
#include "speaker/speaker_layout_lookup.h"
#include "speaker/speaker_trim.h"

namespace joc::speaker {

// One metadata target matrix: object_count x layout channel_count in the
// renderer's internal channel order, before the ramp/timing stage.
struct MetadataTargets {
    std::uint32_t object_count = 0;
    std::uint32_t channel_count = 0;
    std::vector<double> values;  // object major

    double at(std::uint32_t object, std::uint32_t channel) const {
        return values[static_cast<std::size_t>(object) * channel_count + channel];
    }
};

// Computes the metadata target matrix of one block of a parsed frame without
// touching renderer state: the reference's _init_meta validation plus
// _block_targets.  ``trim`` carries the current programme trim (bypass flags),
// ``controls`` the derived per-layout controls, ``size_budget`` the priority
// budget (0 means unlimited, the production value).
Status metadata_block_targets(const LayoutInfo& layout, const oamd::OamdUpdate& update,
                              std::uint32_t block_index,
                              const ejoc_speaker_controls& controls,
                              const oamd::OamdTrim* trim, std::uint32_t size_budget,
                              MetadataTargets* out);

// Stateful metadata-path renderer.  It owns the object target ramps and the
// cross-block pending schedule; the legacy point-renderer state stays in the
// C ABI renderer, which is used for the blocks of a frame that precede the first
// metadata targets (the reference renders those with its legacy block path).
class MetaRenderer {
public:
    MetaRenderer();
    ~MetaRenderer();
    MetaRenderer(MetaRenderer&&) noexcept;
    MetaRenderer& operator=(MetaRenderer&&) noexcept;
    MetaRenderer(const MetaRenderer&) = delete;
    MetaRenderer& operator=(const MetaRenderer&) = delete;

    // Binds the target layout.  Must be called once before process_frame().
    Status configure(const LayoutInfo& layout);

    // Drops the ramp, schedule and metadata state (the layout stays bound).
    Status reset();

    bool configured() const;
    // True once a metadata frame has been accepted (the reference's _meta_mode).
    bool engaged() const;

    const char* last_error() const;

    Status process_frame(ejoc_speaker_renderer_handle legacy,
                         const float* objects16_interleaved, std::uint32_t sample_count,
                         const oamd::OamdUpdate* update, std::uint32_t metadata_offset,
                         const ejoc_speaker_controls& controls, const oamd::OamdTrim* trim,
                         std::uint32_t raw_trim_mode, std::uint32_t size_budget,
                         double* output_interleaved);

    // Computes the target matrix of one block of ``update`` into ``out`` using a
    // scratch context; unlike metadata_block_targets() this validates through
    // the renderer's own state-independent path.
    Status block_targets(const oamd::OamdUpdate& update, std::uint32_t block_index,
                         const ejoc_speaker_controls& controls, const oamd::OamdTrim* trim,
                         std::uint32_t size_budget, MetadataTargets* out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace joc::speaker
