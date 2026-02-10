#pragma once
#include <cstdint>

#define TT_8X(v) v, v, v, v, v, v, v, v
#define TT_16X(v) TT_8X(v), TT_8X(v)
#define TT_32X(v) TT_16X(v), TT_16X(v)

constexpr uint8_t pack_tile_num_faces[32] = {TT_32X(4)};
constexpr uint8_t pack_partial_face[32] = {TT_32X(0)};
constexpr uint8_t pack_tile_face_r_dim[32] = {TT_32X(16)};
constexpr uint8_t pack_narrow_tile[32] = {TT_32X(0)};
constexpr uint8_t pack_tile_r_dim[32] = {TT_32X(32)};
constexpr uint8_t pack_tile_c_dim[32] = {TT_32X(32)};
constexpr uint16_t pack_tile_size[32] = {TT_32X(2048)};

#undef TT_32X
#undef TT_16X
#undef TT_8X
