#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h> 
#include "tensor.h"

float min(float x, float y)
{
    if (x < y)
        return x;
    return y;
}

float max(float x, float y)
{
    if (x > y)
        return x;
    return y;
}

float area(float *f)
{
    float xmin = f[0];
    float ymin = f[1];
    float xmax = f[2];
    float ymax = f[3];
    float rv = max(xmax - xmin, 0) * max(ymax - ymin, 0);
    return rv;
}

float get_iou(float *f1, float *f2)
{
    const float eps = 1e-5;

    float xmin = max(f1[0], f2[0]);
    float ymin = max(f1[1], f2[1]);
    float xmax = min(f1[2], f2[2]);
    float ymax = min(f1[3], f2[3]);
    float rv = max(xmax - xmin, 0) * max(ymax - ymin, 0);
    float s = area(f1) + area(f2);

    rv = rv / max(s - rv, eps);
    return rv;
}

Box *create_box(float *f)
{
    assert(f != NULL);

    Box *box = malloc(sizeof(Box));
    box->xmin = f[0];
    box->ymin = f[1];
    box->xmax = f[2];
    box->ymax = f[3];
    box->score = f[4];
    return box;
}

void destroy_box(Box *box)
{
    assert(box != NULL);

    free(box);
}

Boxes *create_boxes(Box *box, Boxes *boxes)
{
    assert(box != NULL);

    Boxes *rv = malloc(sizeof(Boxes));
    rv->box = box;
    rv->next = boxes;
    return rv;
}

void destroy_boxes(Boxes *boxes)
{
    Box *box;
    Boxes *pre;
    while (boxes)
    {
        box = boxes->box;
        pre = boxes;
        boxes = boxes->next;
        destroy_box(box);
        free(pre);
    }
}

Boxes *nms(Tensor *dst, float confidence, float threshold)
{
    // step1: to box tensor
    // step2: nms and conf
    // step3: get boxes
    assert(dst != NULL);
    // assume input image size: 512x512x3
    // dst feature map: 64x64x25
    // step: 8
    int h = dst->height;
    int w = dst->width;
    int c1 = dst->channel; // assume 25
    int c2 = 5;
    int wc1 = w * c1, wc2 = w * c2;
    Tensor *tensor = create_tensor(dst->height, dst->width, 5);
    float cx = 0.0;
    float cy = 0.0;
    float tw = 0.0;
    float th = 0.0;
    float score = -1.0;
    float tmp = 0.0;

    // step1: active forward to boxes
    // anchors
    const float anchorW[] = {8.0, 16.0, 24.0, 22.0, 12.0};
    const float anchorH[] = {12.0, 22.0, 24.0, 16.0, 8.0};
    float aw, ah;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            // score is already actived
            // active center bias
            score = -.3;
            for (int k = 0; k < c1; k += 5)
            {
                tmp = dst->data[i * wc1 + j * c1 + k + 4];
                if (tmp > score)
                {
                    score = tmp;
                    cx = dst->data[i * wc1 + j * c1 + k];
                    cy = dst->data[i * wc1 + j * c1 + k + 1];
                    tw = dst->data[i * wc1 + j * c1 + k + 2];
                    th = dst->data[i * wc1 + j * c1 + k + 3];
                    aw = anchorW[k / 5];
                    ah = anchorH[k / 5];
                }
            }
            tensor->data[i * wc2 + j * c2 + 4] = score;
            if (score < confidence)
                continue;
            cx = (cx + j) * 8.0;
            cy = (cy + i) * 8.0;
            tw = exp(tw) * aw;
            th = exp(th) * ah;
            tensor->data[i * wc2 + j * c2] = cx - tw / 2;
            tensor->data[i * wc2 + j * c2 + 1] = cy - th / 2;
            tensor->data[i * wc2 + j * c2 + 2] = cx + tw / 2;
            tensor->data[i * wc2 + j * c2 + 3] = cy + th / 2;
        }
    destroy_tensor(dst);

    // step2: nms and conf
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            // first utils confidence
            float s1 = tensor->data[i * wc2 + j * c2 + 4];
            if (s1 < confidence)
            {
                tensor->data[i * wc2 + j * c2 + 4] = -.3;
                continue;
            }

            // nms search at round 9x9 area
            for (int ii = i - 4; ii < i + 5; ii++)
                for (int jj = j - 4; jj < j + 5; jj++)
                {
                    if ((i == ii) && (j == jj))
                        continue;
                    if ((ii < 0) || (jj < 0) || (ii >= h) || (jj >= w))
                        continue;
                    // if some where confdience larger than this position?
                    float s2 = tensor->data[ii * wc2 + jj * c2 + 4];
                    if (s2 < confidence)
                        continue;
                    if (s2 < s1)
                        continue;
                    // calculate the overlap
                    float iou = get_iou(&(tensor->data[i * wc2 + j * c2]), &(tensor->data[ii * wc2 + jj * c2]));
                    if (iou < threshold)
                        continue;
                    tensor->data[i * wc2 + j * c2 + 4] = -.3;
                    ii = i + 5;
                    jj = j + 5;
                }
        }
    destroy_tensor(tensor);

    // step3: collect all boxes to a stack
    Boxes *boxes = NULL; // a stack structure
    int n = 0;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            score = tensor->data[i * wc2 + j * c2 + 4];
            if (score < confidence)
                continue; 
            Box *box = create_box(&(tensor->data[i * wc2 + j * c2]));
            boxes = create_boxes(box, boxes);
            n += 1;
        }
    printf(" %d boxes was detected.\n", n);
    return boxes;
}

Tensor *image2tensor(unsigned char *data, int w, int h, int c)
{
    Tensor *tensor = create_tensor(h, w, 3);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            if (c >= 3)
            { // r,g,b
                for (int k = 0; k < c; k++)
                    tensor->data[i * w * 3 + j * 3 + k] = (1.0 * data[i * w * c + j * c + k]) / 255.0;
            }
            else
            { // seem it as gray
                for (int k = 0; k < c; k++)
                    tensor->data[i * w * 3 + j * 3 + k] = (1.0 * data[i * w * c + j * c]) / 255.0;
            }
        }
    return tensor;
}

Tensor *gsd_resample(Tensor *src, float gsd)
{
    assert(src != NULL);

    float f;
    int h, w;

    f = (gsd / 12.5) * (src->height);
    f = ceil(f / 32);
    h = 32 * (int)f;
    f = (gsd / 12.5) * (src->width);
    f = ceil(f / 32);
    w = 32 * (int)f;
    Tensor *dst = create_tensor(h, w, src->channel);
    int c = src->channel;
    printf("image is resampled to %dx%d, inference starting ...\n", h, w);

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            f = ((float)i) / ((float)h);
            f = f * src->height;
            f = round(f);
            int ii = (int)f;
            if (ii < 0)
                ii = 0;
            if (ii >= src->height)
                ii = src->height - 1;
            f = ((float)j) / ((float)h);
            f = f * src->width;
            f = round(f);
            int jj = (int)f;
            if (jj < 0)
                jj = 0;
            if (jj >= src->width)
                jj = src->width - 1;
            for (int k = 0; k < c; k++)
                dst->data[i * w * c + j * c + k] = src->data[ii * c * src->width + jj * c + k];
        }
    destroy_tensor(src);
    return dst;
}

int float2int(float f, int left, int right)
{
    f = round(f);
    int rv = (int)f;
    if (rv < left)
        rv = left;
    if (rv >= right)
        rv = right - 1;
    return rv;
}

int draw_red_line(unsigned char *data, int h, int w, int c, int xmin, int xmax, int y)
{
    assert(data != NULL);
    if ((y < 0) || (y >= h))
        return 0;
    int i = y;
    for (int j = xmin; j < xmax; j++)
        data[i * w * c + j * c] = 255;
}

int draw_red_column(unsigned char *data, int h, int w, int c, int ymin, int ymax, int x)
{
    assert(data != NULL);
    if ((x < 0) || (x >= w))
        return 0;
    int j = x;
    for (int i = ymin; i < ymax; i++)
        data[i * w * c + j * c] = 255;
    return 0;
}

void visual(unsigned char *data, int w, int h, int c, Boxes *boxes, float sw, float sh)
{
    // sw: width resize scale
    // sh: height resize scale
    // draw a box for each predict
    assert(data != NULL);

    while (boxes != NULL)
    {
        Box *box = boxes->box;
        boxes = boxes->next;

        int xmin = float2int(sw * box->xmin, 0, w);
        int xmax = float2int(sw * box->xmax, 0, w);
        int ymin = float2int(sh * box->ymin, 0, h);
        int ymax = float2int(sh * box->ymax, 0, h);
        int i, j;
        // draw top line
        draw_red_line(data, h, w, c, xmin, xmax, ymin);
        draw_red_line(data, h, w, c, xmin, xmax, ymin - 1);
        draw_red_line(data, h, w, c, xmin, xmax, ymin + 1);
        // draw bottom line
        draw_red_line(data, h, w, c, xmin, xmax, ymax);
        draw_red_line(data, h, w, c, xmin, xmax, ymax - 1);
        draw_red_line(data, h, w, c, xmin, xmax, ymax + 1);
        // draw left line
        draw_red_column(data, h, w, c, ymin, ymax, xmin);
        draw_red_column(data, h, w, c, ymin, ymax, xmin - 1);
        draw_red_column(data, h, w, c, ymin, ymax, xmin + 1);
        // draw right line
        draw_red_column(data, h, w, c, ymin, ymax, xmax);
        draw_red_column(data, h, w, c, ymin, ymax, xmax - 1);
        draw_red_column(data, h, w, c, ymin, ymax, xmax + 1);
    }
}

int save2txt(char *save_name, Boxes *boxes, float sw, float sh)
{
    // sw: width resize scale
    // sh: height resize scale 
    FILE *fp = fopen(save_name, "w");
    if (fp == NULL)
    {
        printf(" unable to create file: %s\n", save_name);
        return -1;
    }
    fprintf(fp, "score xmin ymin xmax ymax\n");
    // save all predict to txt file
    while (boxes != NULL)
    {
        Box *box = boxes->box;
        boxes = boxes->next;

        float xmin = sw * box->xmin;
        float xmax = sw * box->xmax;
        float ymin = sh * box->ymin;
        float ymax = sh * box->ymax;
        float score = box->score;

        fprintf(fp, "%.2f %.2f %.2f %.2f %.2f\n", score, xmin, ymin, xmax, ymax);
    }
    fclose(fp);
    return 0;
}

char *get_save_name(char *img_name)
{
    char *save_name = malloc(sizeof(char) * 1024);
    assert(save_name);
    const char prefix[] = {'r', 'e', 's', 'u', 'l', 't', '/'};
    int n = 0;
    while (img_name[n] != '\0')
        n++;
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        if (img_name[i] == '/')
            k = i + 1;
    }
    for (int i = 0; i < 7; i++)
    {
        save_name[i] = prefix[i];
    }
    for (int i = 0; i + k < n; i++)
    {
        save_name[7 + i] = img_name[i + k];
    }
    save_name[7 + n - k] = '\0';
    return save_name;
}

char *get_txt_name(char *img_name)
{
    char *tmp = get_save_name(img_name);
    char *txt_name = malloc(sizeof(char) * 1024);
    assert(txt_name);

    int n = 0;
    while (tmp[n] != '\0')
        n++;
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        if (tmp[i] == '.')
            k = i + 1;
    }
    for (int i = 0; i < k; i++)
    {
        txt_name[i] = tmp[i];
    }
    free(tmp);
    txt_name[k] = 't';
    txt_name[k + 1] = 'x';
    txt_name[k + 2] = 't';
    txt_name[k + 3] = '\0';
    return txt_name;
}
