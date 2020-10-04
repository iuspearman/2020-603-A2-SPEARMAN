//============================================================================
// Name        : HelloWorld.cpp
// Author      : Isaiah Spearman
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include<cuda_runtime.h>

using namespace std;


__global__ void KNNGPU(int num_attributes, int num_instances, int k_num, float *dataset, int *class_arr, int *predictions)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_instances)
    {
    	float* dists = (float*)malloc(k_num * sizeof(float));
    	int* neighs = (int*)malloc(k_num * sizeof(int));
    	int* sdc_arr = (int*)malloc(k_num * sizeof(int));
    	int max_dist_index = 0;
		int min_dist_index = 0;

    	for(int a = 0; a < k_num; a++)
		{
			dists[a] = FLT_MAX;
		}

        for (int j = 0; j < num_instances; j++)
        {
            if (i == j) { continue; }

            float distance = 0;

            for (int k = 0; k < num_attributes; k++)
            {
                int diff = dataset[i * num_attributes + k] - dataset[j * num_attributes + k];
                distance += diff * diff;
            }

            for(int l = 0; l < k_num; l++)
			{
				if(dists[l] > dists[max_dist_index])
				{
					max_dist_index = l;
				}
			}
            distance = sqrt(distance);
			if(distance < dists[max_dist_index]) // select the closest one
			{
				dists[max_dist_index] = distance;
				neighs[max_dist_index] = j;
				sdc_arr[max_dist_index] = class_arr[j];
			}
        }

        int max_count = 0;
		int sdc;
		for(int b = 0; b < k_num; b++)
		{
			int count = 0;
			for(int c = 0; c < k_num; c++)
			{
				if(sdc_arr[b]==sdc_arr[c])
					count++;
			}
			if(count > max_count)
			{
				max_count = count;
				sdc = sdc_arr[b];
			}
		}

		predictions[i] = sdc;
    }
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    //printf("ccm\n");
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        //printf("%d\n", i);
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		cout << "Usage: ./main datasets/datasetFile.arff K" << endl;
		exit(0);
	}

	ArffParser parser(argv[1]);
	ArffData *dataset = parser.parse();

	struct timespec start, end;
	uint64_t diff;

    const int num_attributes = dataset->num_attributes() - 1;
    const int num_instances = dataset->num_instances();


    int* actual_classes_host;
    int* actual_classes_device;
    int* predictions_device;
    float* data_instances_device;
    float* data_instances_host;

    int* h_A = (int*)malloc(dataset->num_instances() * sizeof(int));
    cudaMalloc((void**)&data_instances_device, num_attributes * num_instances * sizeof(float));
    cudaMalloc((void**)&actual_classes_device, num_instances * sizeof(int));
    cudaMalloc((void**)&predictions_device, num_instances * sizeof(int));

    cudaMallocHost((void**)&data_instances_host, num_attributes * num_instances * sizeof(float));
    cudaMallocHost((void**)&actual_classes_host, num_instances * sizeof(int));

    for (int i = 0; i < num_instances; i++)
    {
        actual_classes_host[i] = dataset->get_instance(i)->get(num_attributes)->operator int32();
        for (int j = 0; j < num_attributes; j++)
        {
            data_instances_host[i * num_attributes + j] = dataset->get_instance(i)->get(j)->operator float();
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    cudaMemcpy((void*)data_instances_device, (void*)data_instances_host, num_attributes * num_instances * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)actual_classes_device, (void*)actual_classes_host, num_instances * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (num_instances + blockSize - 1) / blockSize;

    KNNGPU <<< blockSize, gridSize >>> (num_attributes, num_instances, stoi(argv[2]), data_instances_device, actual_classes_device, predictions_device);

    cudaMemcpy((void*)h_A, (void*)predictions_device, num_instances * sizeof(int), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

	int* confusionMatrix = computeConfusionMatrix(h_A, dataset);
	float accuracy = computeAccuracy(confusionMatrix, dataset);
	printf("The %sNN classifier sequential for %lu instances required %llu ms CPU time, accuracy was %.4f\n", argv[2], dataset->num_instances(), (long long unsigned int) diff, accuracy);

	cudaFree(data_instances_device);
	cudaFree(predictions_device);
	cudaFree(actual_classes_device);
	cudaFree(actual_classes_host);
	cudaFree(data_instances_host);
	cudaFree(h_A);

	return 0;
}
