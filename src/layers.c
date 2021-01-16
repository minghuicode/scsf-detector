#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <math.h>

#include "tensor.h" 
 

Feat *create_feature(int height, int width)
{
    Feat *feat = malloc(sizeof(Feat));
    assert(feat != NULL);

    feat->height = height;
    feat->width = width;
    feat->data = malloc(sizeof(float) * height * width);
    assert(feat->data != NULL);
    return feat;
}

void destroy_feature(Feat *feat)
{
    assert(feat != NULL);
    assert(feat->data != NULL);

    free(feat->data);
    free(feat);
}

Tensor *create_tensor(int height, int width, int channel)
{
    Tensor *tensor = malloc(sizeof(Tensor));
    assert(tensor != NULL);

    tensor->channel = channel;
    tensor->height = height;
    tensor->width = width;
    tensor->data = malloc(sizeof(float) * height * width * channel);
    assert(tensor->data != NULL);

    return tensor;
}

void print_tensor(Tensor *tensor)
{
    printf(" height: %d, width: %d, channel: %d\n", tensor->height, tensor->width, tensor->channel);
}
void print_active(Active active)
{
    if (active == batchNorm)
        printf(" active is BatchNorm %d\n", batchNorm);
    else
        printf(" active is biasAdd %d\n", biasAdd);
}

void destroy_tensor(Tensor *tensor)
{
    assert(tensor != NULL);
    assert(tensor->data != NULL);

    free(tensor->data);
    free(tensor);
}

Tensor *up(Tensor *input)
{
    // upsample x2
    assert(input != NULL);
    Tensor *output = create_tensor(2 * input->height, 2 * input->width, input->channel);

    int h1 = input->height, h2 = output->height;
    int w1 = input->width, w2 = output->width;
    int c = input->channel;
    int i, j, x, y, k;
    for (i = 0; i < h1; i++)
        for (j = 0; j < w1; j++)
        {
            x = 2 * i;
            y = 2 * j;
            memcpy(&(output->data[x*w2*c+y*c]),&(input->data[i*w1*c+j*c]),sizeof(float)*c); 
            y = 1 + 2 * j;
            memcpy(&(output->data[x*w2*c+y*c]),&(input->data[i*w1*c+j*c]),sizeof(float)*c); 
            x = 1 + 2 * i;
            memcpy(&(output->data[x*w2*c+y*c]),&(input->data[i*w1*c+j*c]),sizeof(float)*c); 
            y = 2 * j;
            memcpy(&(output->data[x*w2*c+y*c]),&(input->data[i*w1*c+j*c]),sizeof(float)*c);  
        }
    return output;
}

void sigmoid(Tensor *src)
{
    // in-place sigmoid
    assert(src != NULL);

    int h = src->height;
    int w = src->width;
    int c = src->channel;
    int wc = w * c;

    if (c == 1) // active score layer
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                src->data[i * wc + j * c] = 1 / (1 + exp(-src->data[i * wc + j * c]));
    }
    else // c is 25, active regress layer
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int k = 0; k < c; k += 5)
                {
                    src->data[i * wc + j * c + k + 4] = 1 / (1 + exp(-src->data[i * wc + j * c + k + 4])); // score
                    src->data[i * wc + j * c + k] = 1 / (1 + exp(-src->data[i * wc + j * c + k]));         // cx
                    src->data[i * wc + j * c + k + 1] = 1 / (1 + exp(-src->data[i * wc + j * c + k + 1])); // cy
                }
    }
}

void multiply(Tensor *dst, Tensor *src)
{
    // in-place multiply
    assert(dst != NULL);
    assert(src != NULL);
    assert(dst->height == src->height);
    assert(dst->width == src->width);

    int h = src->height;
    int w = src->width;
    int c1 = src->channel;
    int c2 = dst->channel;
    int wc1 = w * c1, wc2 = w * c2; 
    if (c2 == 1) // score and score
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                dst->data[i * wc2 + j * c2] *= src->data[i * wc1 + j * c1];
    }
    else //c2 is 25, socre and regression
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int k = 0; k < c2; k += 5)
                    dst->data[i * wc2 + j * c2 + k + 4] *= src->data[i * wc1 + j * c1];
    }
}

Tensor *concat(Tensor *t1, Tensor *t0)
{
    Tensor *t2 = up(t0);

    assert(t1 != NULL);
    assert(t2 != NULL);
    assert(t1->height == t2->height);
    assert(t1->width == t2->width);

    Tensor *dst = create_tensor(t1->height, t1->width, t1->channel + t2->channel);
    int h = dst->height;
    int w = dst->width;
    int c = dst->channel;
    int c1 = t1->channel, c2 = t2->channel;
    int wc = w * c;
    int wc1 = w * c1, wc2 = w * c2; 

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            memcpy(&(dst->data[i * wc + j * c]), &(t1->data[i * wc1 + j * c1]), sizeof(float) * c1);
            memcpy(&(dst->data[i * wc + j * c + c1]), &(t2->data[i * wc2 + j * c2]), sizeof(float) * c2);
        }
    destroy_tensor(t2);
    return dst;
}

float mean(Feat *feat)
{
    int h = feat->height;
    int w = feat->width;
    register float s = 0;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            s += feat->data[i * w + j];
    s = s / (1.0 * h * w);
    return s;
}

float var(Feat *feat, float m)
{
    int h = feat->height;
    int w = feat->width;
    register float s = 0;
    register float t;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            t = feat->data[i * w + j] - m;
            s += t * t / (1.0 * h * w);
        }
    return s;
}

// float sqrt(float x)
// {
//     // John Carmack's magic number
//     float xhalf = 0.5f * x;

//     int i = *(int *)&x;
//     i = 0x5f3759df - (i >> 1);
//     x = *(float *)&i;
//     x = x * (1.5f - xhalf * x * x); // Newton step
//     x = x * (1.5f - xhalf * x * x); // Newton step
//     x = x * (1.5f - xhalf * x * x); // Newton step
//     return (1 / x);
// }

void active_bn(Feat *feat, float g, float b)
{
    int h = feat->height;
    int w = feat->width;
    float m, v;
    float eps = 1e-5;
    register float s;
    register float t;

    m = mean(feat);
    v = var(feat, m);
    v = sqrt(v + eps);
    t = g / v;

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            s = feat->data[i * w + j] - m;
            s = s * t + b;
            // leaky relu
            if (s > 0)
                feat->data[i * w + j] = s;
            else
                feat->data[i * w + j] = 0.1 * s;
        }
}

void active_bias(Feat *feat, float b)
{
    int h = feat->height;
    int w = feat->width;

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            feat->data[i * w + j] += b;
}

void feat2tensor(Tensor *tensor, Feat **feats, bool residual)
{
    // combine multi 2-d features to a single 3-d tensor
    assert(tensor != NULL);
    assert(feats != NULL);
    for (int i = 0; i < tensor->channel; i++)
    {
        assert(feats[i] != NULL);
        assert(feats[i]->height == tensor->height);
        assert(feats[i]->width == tensor->width);
        assert(feats[i]->data != NULL);
    }

    int h = tensor->height;
    int w = tensor->width;
    int c = tensor->channel;
    int wc = w * c;
    if (residual == true)
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int k = 0; k < c; k++)
                    tensor->data[i * wc + j * c + k] += feats[k]->data[i * w + j];
    }
    else
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int k = 0; k < c; k++)
                    tensor->data[i * wc + j * c + k] = feats[k]->data[i * w + j];
    }
}

void conv_feat(Feat *feat, Tensor *tensor, float *weight)
{
    // convolute input tensor to a 2-dim feature map
    assert(feat != NULL);
    assert(tensor != NULL);
    assert(feat->height == tensor->height);
    assert(feat->width == tensor->width);
    assert(feat->data != NULL);
    assert(tensor->data != NULL);

    int h = feat->height;
    int w = feat->width;
    int c = tensor->channel;
    int wc = w * c;
    register float s;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            s = 0.0;
            for (int k = 0; k < c; k++)
                s += weight[k] * tensor->data[i * wc + j * c + k];
            feat->data[i * w + j] = s;
        }
}

void conv_active(Feat *feat, Tensor *tensor, Active active, float *weight, float gamma, float bias)
{
    // conv and active
    conv_feat(feat, tensor, weight);
    if (active == batchNorm)
        active_bn(feat, gamma, bias);
    else
        active_bias(feat, bias);
}

/********************************************* pthread + *******************************************/

Param *create_param(Feat *feat, Tensor *tensor, Conv *conv, int k)
{
    // param for pthread
    Param *param = malloc(sizeof(Param));
    assert(param != NULL);

    param->feat = feat;
    param->tensor = tensor;
    param->conv = conv;
    param->k = k;
    return param;
}

void destroy_param(Param *param)
{
    assert(param != NULL);

    free(param);
}

void *thread_func(void *arg)
{
    Param *param = (Param *)arg;
    Conv *cc = param->conv;
    int n = cc->size * cc->size * cc->c_i;
    float *weight = cc->weight;
    float *gamma = cc->gamma;
    float *bias = cc->bias;
    int k = param->k;

    conv_active(param->feat, param->tensor, param->conv->active, &(weight[n * k]), gamma[k], bias[k]);
}

/********************************************* pthread - *******************************************/

void conv1x1(Tensor *dst, Tensor *src, Conv *cc, bool residual)
{
    assert(dst != NULL);
    assert(src != NULL);
    assert(dst->data != NULL);
    assert(src->data != NULL);
    assert(src->channel == cc->size * cc->size * cc->c_i);
    assert(dst->channel == cc->c_o);

    int n = cc->c_o;
    Feat **feats = malloc(sizeof(Feat *) * n);
    Param **params = malloc(sizeof(Param *) * n);
    pthread_t *threads = malloc(sizeof(pthread_t) * n);
    int rv;
    void *retval;
    // start threads
    for (int i = 0; i < n; i++)
    {
        feats[i] = create_feature(dst->height, dst->width);
        params[i] = create_param(feats[i], src, cc, i);
        rv = pthread_create(&(threads[i]), NULL, thread_func, (void *)(params[i]));
        assert(rv == 0);
    }
    // collect threads
    for (int i = 0; i < n; i++)
    {
        rv = pthread_join(threads[i], &retval);
        assert(rv == 0);
    }
    feat2tensor(dst, feats, residual);
    // recovery memory
    for (int i = 0; i < n; i++)
    {
        destroy_param(params[i]);
        destroy_feature(feats[i]);
    }
    free(feats);
    free(threads);
    free(params);
}

Tensor *tensor_reshape(Tensor *src, int stride)
{
    // conv size is 3
    if (stride > 1)
    { // seem it to be 2
        Tensor *dst = create_tensor(src->height / 2, src->width / 2, 9 * src->channel);
        int h1 = src->height, h2 = dst->height;
        int w1 = src->width, w2 = dst->width;
        int c1 = src->channel, c2 = dst->channel;
        int wc1 = w1 * c1, wc2 = w2 * c2;

        for (int x = -1; x < 2; x++)
            for (int y = -1; y < 2; y++)
                for (int i = 0; i < h2; i++)
                    for (int j = 0; j < w2; j++)
                    {
                        int ii = 2 * i + x;
                        int jj = 2 * j + y;
                        if ((ii < 0) || (jj < 0) || (ii >= h1) || (jj >= w1))
                            memset(&(dst->data[i * wc2 + j * c2 + ((x + 1) * 3 + (y + 1)) * c1]), 0, sizeof(float) * c1);
                        else
                            memcpy(&(dst->data[i * wc2 + j * c2 + ((x + 1) * 3 + (y + 1)) * c1]), &(src->data[ii * wc1 + jj * c1]), sizeof(float) * c1);
                    }
        return dst;
    }
    else
    {
        Tensor *dst = create_tensor(src->height, src->width, 9 * src->channel);
        int h = src->height;
        int w = dst->width;
        int c1 = src->channel, c2 = dst->channel;
        int wc1 = w * c1, wc2 = w * c2;

        for (int x = -1; x < 2; x++)
            for (int y = -1; y < 2; y++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                    {
                        int ii = i + x;
                        int jj = j + y;
                        if ((ii < 0) || (jj < 0) || (ii >= h) || (jj >= w))
                            memset(&(dst->data[i * wc2 + j * c2 + ((x + 1) * 3 + (y + 1)) * c1]), 0, sizeof(float) * c1);
                        else
                            memcpy(&(dst->data[i * wc2 + j * c2 + ((x + 1) * 3 + (y + 1)) * c1]), &(src->data[ii * wc1 + jj * c1]), sizeof(float) * c1);
                    }
        return dst;
    }
}

void conv3x3(Tensor *dst, Tensor *src, Conv *cc, bool residual)
{
    // reshape -> conv1x1 -> dst
    Tensor *tensor = tensor_reshape(src, cc->stride);
    conv1x1(dst, tensor, cc, residual);
    destroy_tensor(tensor);
}

void conv_layer(Tensor *dst, Tensor *src, Conv *cc, bool residual)
{
    if (cc->size == 1)
        conv1x1(dst, src, cc, residual);
    else
        conv3x3(dst, src, cc, residual);
}

Tensor *twin_layer(Tensor *src, Twin *twin)
{
    assert(src != NULL);
    assert(twin != NULL);
    assert(src->channel == twin->c_i);

    Tensor *tmp = create_tensor(src->height, src->width, twin->c_t);
    Tensor *dst;
    if (twin->residual == true)
        dst = src;
    else
    {
        if (twin->st2 == 1)
            dst = create_tensor(src->height, src->width, twin->c_o);
        else
            dst = create_tensor(src->height / 2, src->width / 2, twin->c_o);
    }
    // forward
    conv_layer(tmp, src, twin->c1, false);
    conv_layer(dst, tmp, twin->c2, twin->residual);
    destroy_tensor(tmp);
    return dst;
}

Tensor *stage_layer(Tensor *src, Stage *stage)
{
    assert(src != NULL);
    assert(stage != NULL);

    Tensor *tmp;
    Tensor *tensor = twin_layer(src, stage->twins[0]);

    for (int i = 1; i < stage->num; i++)
    {
        tmp = twin_layer(tensor, stage->twins[i]);
        if (stage->twins[i]->residual == false)
            destroy_tensor(tensor); 
        tensor = tmp;
    }
    return tensor;
}

Tensor *forward(Tensor *src, Network *network)
{
    // check -> forward -> head
    // check input src
    assert(network != NULL);
    assert(src != NULL);
    assert(src->height);
    assert(src->height % 32 == 0);
    assert(src->width);
    assert(src->width % 32 == 0);
    assert(src->channel == 3);

    // model forward
    Tensor *tmp;
    Tensor *tensor = src;
    Tensor *A, *B, *C, *H, *L;
    Tensor *leaf1, *leaf2, *fruit;
 
    // component 1 of 9:
    tmp = stage_layer(tensor, network->root);
    destroy_tensor(tensor);
    tensor = tmp;
    // component 2 of 9:
    A = stage_layer(tensor, network->trunkA);
    destroy_tensor(tensor);
    tensor = A;
    // component 3 of 9:
    B = stage_layer(tensor, network->trunkB);
    tensor = B;
    // component 4 of 9:
    C = stage_layer(tensor, network->trunkC);
    // component 5 of 9:
    tensor = concat(B, C);
    H = stage_layer(tensor, network->brunchH);
    // component 6 of 9:
    destroy_tensor(tensor);
    tensor = concat(A, H);
    L = stage_layer(tensor, network->brunchL);
    destroy_tensor(tensor);
    destroy_tensor(A);
    destroy_tensor(B);
    // component 7 of 9:
    leaf1 = twin_layer(C, network->leafH);
    destroy_tensor(C);
    sigmoid(leaf1); 
    // component 8 of 9:
    leaf2 = twin_layer(H, network->leafL);
    destroy_tensor(H);
    sigmoid(leaf2); 
    // componennt 9 of 9:
    fruit = twin_layer(L, network->fruit);
    destroy_tensor(L);
    sigmoid(fruit);
    // Confidence Feature Pyramid Network
    tmp = up(leaf1);
    destroy_tensor(leaf1);
    leaf1 = tmp;
    multiply(leaf2, leaf1);
    destroy_tensor(leaf1);
    tmp = up(leaf2);
    destroy_tensor(leaf2);
    leaf2 = tmp;
    multiply(fruit, leaf2);
    destroy_tensor(leaf2); 
 
    return fruit;
}