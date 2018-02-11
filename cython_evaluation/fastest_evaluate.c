#include<math.h>

int eval(double* out, int size_out, int x_num, double x_min, double x_max, double h_min, double h_max)
{
    const double dx = (x_max - x_min) / (x_num - 1);

    #pragma omp parallel for
    for(int h_i = 0; h_i < size_out; h_i++){

        const double h = h_min + (h_max - h_min) * h_i / (size_out - 1);

        double result = 0.;

        for(double X = x_min; X < x_max + dx; X += dx)
            for(double Y = x_min; Y < x_max + dx; Y += dx)
                for (double Z = x_min; Z < x_max + dx; Z += dx)
                    result += exp(-pow(2. * Z + X + pow(Y, 2) - h, 2)) * (sin(X + Y + 3. * Z + h) + pow(Y + Z + h, 2));

        out[h_i] = result;
    }

    // success
    return 0;
}