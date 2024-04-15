#include <stdio.h>

struct model{
    double w;
    double b;
};

double mse(double x[], double y[], struct model m){
    int n=5;
    double err = 0.0;
    for(int i=0; i<n; i++){
        double y_p = (m.w)*x[i] + (m.b);
        err += (y_p - y[i]) * (y_p - y[i]);
    }
    return err;
}

double grad_w(double x[], double y[], struct model m){
    int n=5;
    double val = 0.0;
    for(int i=0; i<n; i++){
        double y_p = (m.w)*x[i] + (m.b);
        val += x[i] * (y_p - y[i]);
    }
    return val;
}

double grad_b(double x[], double y[], struct model m){
    int n=5;
    double val = 0.0;
    for(int i=0; i<n; i++){
        double y_p = (m.w)*x[i] + (m.b);
        val += (y_p - y[i]);
    }
    return val;
}

int main(){
    int num;
    FILE *fptr;

    if((fptr = fopen("train.csv", "r")) == NULL){
        printf("Error\n");
    }

    fscanf(fptr, "%d", num);
    printf("%d\n", num);

    double x[5] = {1,2,3,4,5};
    double y[5] = {2,5,8,10,13};

    struct model m;
    m.w = 0.9;
    m.b = 0.3;

    double lr = 0.01;
    int num_epochs = 50;

    for(int i=0; i<num_epochs + 1; i++){
        if(i%5 == 0){
            printf("Error %d: %f\n", i, mse(x,y,m));
        }
        m.w -= lr/5 * grad_w(x,y,m);
        m.b -= lr/5 * grad_b(x,y,m);
    }
}