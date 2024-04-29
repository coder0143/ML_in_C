#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct model{
    double w0;
    double w1;
    double b;
};

double sigmoid(double x){
    return 1/(1+exp(-x));
}

double cost(double x[2][500], double y[500], struct model m, int length){
    double err = 0.0;
    double epsilon = 1e-5;
    for(int i=0; i<length; i++){
        double y_p = sigmoid(m.w0*x[0][i] + m.w1*x[1][i] + m.b);
        err += (1/length)*((y[i] * log(y_p + epsilon)) - ((1-y[i]) * (log(1-y_p + epsilon))));
    }
    return err;
}

double grad_w0(double x[2][500], double y[500], struct model m, int length){
    double grad = 0.0;
    for(int i=0; i<length; i++){
        double y_p = sigmoid(m.w0*x[0][i] + m.w1*x[1][i] + m.b);
        grad += y_p * x[0][i];
    }
    return grad;
}

double grad_w1(double x[2][500], double y[500], struct model m, int length){
    double grad = 0.0;
    for(int i=0; i<length; i++){
        double y_p = sigmoid(m.w0*x[0][i] + m.w1*x[1][i] + m.b);
        grad += y_p * x[1][i];
    }
    return grad;
}

double grad_b(double x[2][500], double y[500], struct model m, int length){
    double grad = 0.0;
    for(int i=0; i<length; i++){
        double y_p = sigmoid(m.w0*x[0][i] + m.w1*x[1][i] + m.b);
        grad += y_p;
    }
    return grad;
}

int main(void){
    // file reading
    char buffer[1000];
    char *data;
    // input array
    double input_arr[2][500];
    // output array
    double output_arr[500];
    // prediction array
    int pred_arr[500];
    
    // file open
    FILE *train = fopen("train.csv", "r"); // dataset name(csv)
    if(train == NULL){
        printf("File not found\n");
        exit(-1);
    }

    // Writing to arrays
    int len=0;
    while(fgets(buffer, sizeof(buffer), train)){
        if(len>0){
            // 1st col
            data = strtok(buffer, ",");
            input_arr[0][len-1] = atof(data);
            // Second col
            data = strtok(NULL, ",");
            input_arr[1][len-1] = atof(data);
            // Third col
            data = strtok(NULL, ",");
            output_arr[len-1] = atof(data);
        }
        len++;
    }

    // display dataset
    int length = len-1;
    // for(int i=0; i < length; i++){
    //     printf("%.2f %.2f %.2f\n", input_arr[0][i], input_arr[1][i], output_arr[i]);
    // }
    
    // initialize model
    struct model m;
    m.w0 = 0.6;
    m.w1 = 0.5;
    m.b = 0.4;

    // training
    double lr = 0.005;
    int n_epochs = 100;

    for(int i=0; i<n_epochs + 1; i++){
        if(i%10 == 0){
            //printf("Error %d: %f\n", i, cost(input_arr,output_arr,m,length));
        }
        double mw0 = (lr/length)*(grad_w0(input_arr,output_arr,m,length));
        double mw1 = (lr/length)*(grad_w1(input_arr,output_arr,m,length));
        double mb = (lr/length)*(grad_b(input_arr,output_arr,m,length));

        m.w0 -= mw0;
        m.w1 -= mw1;
        m.b -= mb;
    }

    // Predictions and accuracy
    for(int i=0; i<length; i++){
        double y_p = sigmoid(m.w0*input_arr[0][i] + m.w1*input_arr[1][i] + m.b);
        if (y_p >= 0.5){
            pred_arr[i] = 1;
        }   
        else{
            pred_arr[i] = 0;
        }
    }

    double right = 0;
    for(int i=0; i<length; i++){
        if (output_arr[i] == pred_arr[i]){
            right += 1.0;
        }
    }

    double acc = right / 500.0;

    printf("Overall accuracy: %f\n", acc);

}
