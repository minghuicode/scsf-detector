#ifndef __TENSOR_H__
#define __TENSOR_H__ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <math.h>

typedef enum
{
  false = 0,
  true = !false
} bool;

typedef enum {
    batchNorm,
    biasAdd,
} Active;

typedef struct {
    float xmin, ymin, xmax, ymax;
    float score;
} Box;

typedef struct Boxes{
    Box* box;
    struct Boxes* next;
} Boxes;

typedef struct {
    int height, width;
    float *data;
} Feat; // 2-dim feature map

typedef struct {
    int height, width, channel;
    float *data;
} Tensor;

typedef struct {
    int size, stride;
    int c_i, c_o;
    Active active;
    float* weight;
    float* gamma;
    float* bias;
} Conv;

typedef struct {
    Feat *feat;
    Tensor *tensor;
    Conv *conv;
    int k;
} Param;

typedef struct {
    int s1, s2;
    int st1, st2;
    int c_i, c_t, c_o;
    Conv *c1, *c2;
    bool residual;
} Twin;

typedef struct {
    int num;
    Twin** twins;
} Stage;

typedef struct {
    Stage *root, *trunkA,
          *trunkB, *trunkC,
          *brunchH, *brunchL;
    Twin  *leafH, *leafL,
          *fruit;
} Network;


Tensor *create_tensor(int height, int width, int channel);
void destroy_tensor(Tensor *tensor);
Network *create_network();
Tensor *forward(Tensor *src, Network *network);
void load_network(Network *layer);
void destroy_network(Network *layer);
Boxes *nms(Tensor *dst, float confidence, float threshold);
void destroy_boxes(Boxes *boxes);
Tensor *image2tensor(unsigned char *data, int w, int h, int c);
Tensor *gsd_resample(Tensor *src, float gsd);
void visual(unsigned char *data, int w, int h, int c, Boxes *boxes, float sw, float sh);

int save2txt(char *save_name, Boxes *boxes, float sw, float sh);
char *get_save_name(char *img_name);
char *get_txt_name(char *img_name);

#endif // tensor.h
