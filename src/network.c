#include <stdlib.h> 
#include <assert.h>
#include "tensor.h"

Conv *create_conv(int size, int stride, int c_i, int c_o, Active active)
{
    Conv *layer = malloc(sizeof(Conv));
    assert(layer != NULL);

    layer->size = size;
    layer->stride = stride;
    layer->c_i = c_i;
    layer->c_o = c_o;
    layer->active = active;
    layer->weight = malloc(sizeof(float)*(size*size*c_i*c_o));
    layer->bias = malloc(sizeof(float)*c_o);
    assert(layer->bias != NULL);
    layer->gamma = malloc(sizeof(float)*c_o);
    assert(layer->gamma != NULL);
    // if (active == biasAdd) then gamma make no sense  
    return layer;
}

void load_conv(Conv *layer,FILE *stream)
{
    int n=layer->size*layer->size*layer->c_i*layer->c_o;

    fread(layer->weight, sizeof(float), n, stream);
    fread(layer->bias, sizeof(float), layer->c_o, stream); 
    if (layer->active == batchNorm)
        fread(layer->gamma, sizeof(float), layer->c_o, stream); 
}

void destroy_conv(Conv *layer)
{
    assert(layer != NULL);

    free(layer->weight);
    free(layer->bias);
    if (layer->active == batchNorm)
        free(layer->gamma);
    free(layer);
}

Twin *create_twin(int s1, int st1, int s2, int st2, int c_i, int c_t, int c_o, bool residual, bool head)
{
    Twin *layer = malloc(sizeof(Twin));
    assert(layer != NULL);

    layer->s1 = s1;
    layer->st1 = st2;
    layer->s2 = s2;
    layer->st2 = st2;
    layer->c_i = c_i;
    layer->c_t = c_t;
    layer->c_o = c_o;
    layer->residual = residual;
    layer->c1 = create_conv(s1, st1, c_i,c_t,batchNorm);
    if (head)
        layer->c2 = create_conv(s2, st2, c_t, c_o, biasAdd);
    else
        layer->c2 = create_conv(s2, st2, c_t, c_o, batchNorm);
    return layer;
}

void load_twin(Twin *layer, FILE *stream)
{
    load_conv(layer->c1, stream);
    load_conv(layer->c2, stream);
}

void destroy_twin(Twin *layer)
{
    assert(layer != NULL);

    destroy_conv(layer->c1);
    destroy_conv(layer->c2);
    free(layer);
}

// feature downsample: twin-F
// confidence downsample: twin-C
// residual twin: twin-R
// brunch twin: twin-B
// header twin: twin-H
Twin *create_twinF(int c_i, int c_o)
{
    return create_twin(3, 1, 3, 2, c_i, 4, c_o, false, false);
}
Twin *create_twinC(int c_i, int c_o)
{
    return create_twin(1, 1, 3, 2, c_i, 4, c_o, false, false);
}
Twin *create_twinR(int c_i)
{
    return create_twin(1, 1,  3, 1, c_i, 4, c_i, true, false);
}
Twin *create_twinB(int c_i, int c_t, int c_o)
{
    return create_twin(1, 1, 3, 1, c_i, c_t, c_o, false, false);
}
Twin *create_twinH(int c_i, int c_t, int c_o)
{
    return create_twin(1, 1, 1, 1, c_i, c_t, c_o, false, true);
}

Stage *create_stage(int n) 
{
    Stage *layer = malloc(sizeof(Stage));
    assert(layer != NULL);

    layer->num = n;
    layer->twins = malloc(sizeof(Twin*)*n);
    assert(layer->twins != NULL);
    return layer;
}

void load_stage(Stage *layer, FILE *stream)
{
    for(int i=0;i<layer->num;i++)
        load_twin(layer->twins[i], stream);
}

void destroy_stage(Stage *layer)
{
    assert(layer != NULL);

    for(int i=0;i<layer->num;i++)
        destroy_twin(layer->twins[i]);
    free(layer);
}

Network *create_network()
{
    Network *layer = malloc(sizeof(Network));
    assert(layer != NULL);

    // component 1 of 9: downsample to 1/4
    layer->root = create_stage(5);
    layer->root->twins[0] = create_twinF(3,16);
    layer->root->twins[1] = create_twinR(16);
    layer->root->twins[2] = create_twinF(16,32);
    layer->root->twins[3] = create_twinR(32);
    layer->root->twins[4] = create_twinR(32);
    // component 2 of 9: downsample to 1/8
    layer->trunkA = create_stage(9);
    layer->trunkA->twins[0] = create_twinF(32,64);
    for(int i=1;i<9;i++)
        layer->trunkA->twins[i] = create_twinR(64);
    // component 3 of 9: downsample to 1/16
    layer->trunkB = create_stage(9);
    layer->trunkB->twins[0] = create_twinC(64,32);
    for(int i=1;i<9;i++)
        layer->trunkB->twins[i] = create_twinR(32);
    // component 4 of 9: downsample to 1/32
    layer->trunkC = create_stage(5);
    layer->trunkC->twins[0] = create_twinC(32,16);
    for(int i=1;i<5;i++)
        layer->trunkC->twins[i] = create_twinR(16);
    // component 5 of 9
    layer->brunchH = create_stage(2);
    layer->brunchH->twins[0] = create_twinB(48,4,32);
    layer->brunchH->twins[1] = create_twinB(32,4,32);
    // component 6 of 9
    layer->brunchL = create_stage(2);
    layer->brunchL->twins[0] = create_twinB(96,16,32);
    layer->brunchL->twins[1] = create_twinB(32,16,32);
    // component 7 of 9: downsample to 1/32 head-1
    layer->leafH = create_twinH(16,4,1);
    // component 8 of 9: downsample to 1/16 head-2
    layer->leafL = create_twinH(32,4,1);
    // component 9 of 9: downsample to 1/8 head-3
    layer->fruit = create_twinH(32,64,25);

    return layer;
}

void load_network(Network *layer)
{
    FILE *fp = fopen("vehicle.weight","rb");
    if (fp == NULL)
    {
        printf("please check weight file path\n");
        assert(0);
    }

    load_stage(layer->root,fp);
    load_stage(layer->trunkA,fp);
    load_stage(layer->trunkB,fp);
    load_stage(layer->trunkC,fp);
    load_stage(layer->brunchH,fp);
    load_stage(layer->brunchL,fp);
    load_twin(layer->leafH,fp);
    load_twin(layer->leafL,fp);
    load_twin(layer->fruit,fp);

    char a,b,c,d;
    fread(&a,sizeof(char),1,fp);
    fread(&b,sizeof(char),1,fp);
    fread(&c,sizeof(char),1,fp);
    fread(&d,sizeof(char),1,fp); 
    fclose(fp);
 
    if ((a=='B') && (b=='U') && (c=='A') && (d=='A'))
    {
        printf("network initial successed!\n");
    }
    else
    {
        printf("network initial failed, please check your weight file!\n");
        assert(0);
    }
}

void destroy_network(Network *layer)
{
    assert(layer != NULL);

    destroy_twin(layer->leafH);
    destroy_twin(layer->leafL);
    destroy_twin(layer->fruit);
    destroy_stage(layer->root);
    destroy_stage(layer->trunkA);
    destroy_stage(layer->trunkB);
    destroy_stage(layer->trunkC);
    destroy_stage(layer->brunchH);
    destroy_stage(layer->brunchL);
}
