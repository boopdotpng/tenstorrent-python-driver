#pragma once

#define TT_8X(v) v, v, v, v, v, v, v, v
#define TT_16X(v) TT_8X(v), TT_8X(v)
#define TT_32X(v) TT_16X(v), TT_16X(v)

constexpr unsigned char pack_src_format[32] = {TT_32X(5)};
constexpr unsigned char pack_dst_format[32] = {TT_32X(5)};

#undef TT_32X
#undef TT_16X
#undef TT_8X
