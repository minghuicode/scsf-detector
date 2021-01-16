#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>
#include "tensor.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

// Tensor *create_tensor(int height, int width, int channel);
// Network *create_network();
// Tensor *forward(Tensor *src, Network *network);

int detect(Network *network, float confidence, float threshold)
{
    int w, h, c;
    char *img_name = malloc(sizeof(char) * 1024);
    char *save_name, *txt_name;
    FILE *fp;
    unsigned char *data;

    // *******************  input path  ************************************
    printf("input image name, process terminated if file not exist\n");
    scanf("%s", img_name);
    fp = fopen(img_name, "r");
    if (fp == NULL)
    {
        printf("file not found:%s\ndetection exit.\n", img_name);
        return 1;
    }
    fclose(fp);
    save_name = get_save_name(img_name);
    fp = fopen(save_name, "r");
    if (fp == NULL)
    {
        fp = fopen(save_name, "w");
    }
    if (fp == NULL)
    {
        printf("Please confirm you have result folder in current dict!\n");
        return 2;
    }
    fclose(fp);
    txt_name = get_txt_name(img_name);

    // *******************  pre process  ***********************************
    data = stbi_load(img_name, &w, &h, &c, 0);
    if (data == NULL)
    {
        printf("image loaf fail, check you image: %s\n", img_name);
        return 3;
    }
    float gsd = 12.5;
    Tensor *src = image2tensor(data, w, h, c);
    printf("input the ground sample distance(gsd: cm/pixel), default 12.5\n");
    scanf("%f", &gsd);
    src = gsd_resample(src, gsd);
    float sw = ((float)w) / ((float)src->width);
    float sh = ((float)h) / ((float)src->height);

    // *******************  model inference  *******************************
    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);
    Tensor *dst = forward(src, network);
    gettimeofday(&t_end, NULL); 
    float time_consumed = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
    printf("inference consume %.2f second(s), and", time_consumed);

    // *******************  post process  **********************************
    Boxes *boxes = nms(dst, confidence, threshold);
    save2txt(txt_name, boxes, sw, sh);
    visual(data, w, h, c, boxes, sw, sh);
    stbi_write_png(save_name, w, h, c, data, w * c);

    destroy_boxes(boxes);
    stbi_image_free(data);
    free(img_name);
    free(save_name);
    free(txt_name);
    return 0;
}

int main(int argc, char argv[])
{
    printf("usage:  ./detect conf nms\n");
    printf(" conf in [0,1] for score threshold\n");
    printf(" nms in [0,1] for nms threshold\n");
    float confidence = 0.9; // box confidence
    float threshold = 0.5;  // non max suppress threshold
    if (argc > 1)
    {
        confidence = (float)argv[1];
        if ((confidence < 0) || (confidence > 1))
        {
            printf("error conf setting, make sure it in [0,1]\n");
        }
    }
    if (argc > 2)
    {
        threshold = (float)argv[1];
        if ((threshold < 0) || (threshold > 1))
        {
            printf("error nms setting, make sure it in [0,1]\n");
        }
    }
    printf("conf is set to be %.2f\n", confidence);
    printf("nms is set to be %.2f\n\n\n", threshold);

    Network *network = create_network();
    load_network(network);

    int rv = detect(network, confidence, threshold);
    while (rv == 0)
        rv = detect(network, confidence, threshold);
    destroy_network(network);

    return 0;
}