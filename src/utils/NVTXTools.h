#ifndef NVTXTOOLS_H
#define NVTXTOOLS_H

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);
bool isNVTXEnabled = false;
void enableNVTX() { isNVTXEnabled = true; }
void disableNVTX() { isNVTXEnabled = false; }
#define PUSH_RANGE(name, cid)                                                  \
  {                                                                            \
    int color_id = cid;                                                        \
    color_id = color_id % num_colors;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                   \
    eventAttrib.version = NVTX_VERSION;                                        \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
    eventAttrib.color = colors[color_id];                                      \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
    eventAttrib.message.ascii = name;                                          \
    if (isNVTXEnabled)                                                         \
      nvtxRangePushEx(&eventAttrib);                                           \
  }
#define POP_RANGE                                                              \
  if (isNVTXEnabled) {                                                         \
    cudaDeviceSynchronize();                                                   \
    nvtxRangePop();                                                            \
  }
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
void enableNVTX() {}
void disableNVTX() {}
#endif

#endif
