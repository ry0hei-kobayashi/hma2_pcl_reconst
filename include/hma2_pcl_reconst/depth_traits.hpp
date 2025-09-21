#pragma once

#include <cmath>
#include <limits>

namespace hma2_pcl_reconst
{

// DepthTraits template declaration
template<typename T> struct DepthTraits;

// --- uint16_t depth image (16UC1, mm) ---
template<>
struct DepthTraits<uint16_t>
{
  static inline bool valid(uint16_t depth) { return depth != 0; }
  static inline float toMeters(uint16_t depth) { return depth * 0.001f; }  // mm → m
  static inline uint16_t fromMeters(float depth) { return static_cast<uint16_t>(depth * 1000.0f); }
};

// --- float depth image (32FC1, m) ---
template<>
struct DepthTraits<float>
{
  static inline bool valid(float depth) { return std::isfinite(depth); }
  static inline float toMeters(float depth) { return depth; }
  static inline float fromMeters(float depth) { return depth; }
};

}  // namespace hma2_pcl_reconst

