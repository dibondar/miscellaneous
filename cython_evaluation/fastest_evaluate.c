#include<math.h>

int eval(double* out, int size_out, int x_num, double x_min, double x_max, double h_min, double h_max)
{
    #pragma omp parallel for
    for(int h_i = 0; h_i < size_out; h_i++){

        const double h = h_min + (h_max - h_min) * h_i / (size_out - 1);

        double result = 0.;

        for(int x_i = 0; x_i < x_num; x_i++){
            const double X = x_min + (x_max - x_min) * x_i / (x_num - 1);

            for(int y_i = 0; y_i < x_num; y_i++){
                const double Y = x_min + (x_max - x_min) * y_i / (x_num - 1);

                for (int z_i = 0; z_i < x_num; z_i++) {
                    const double Z = x_min + (x_max - x_min) * z_i / (x_num - 1);

                    result += exp(-pow(2. * Z + X + pow(Y, 2) - h, 2)) * (sin(X + Y + 3. * Z + h) + pow(Y + Z + h, 2));
                }
            }

        }

        out[h_i] = result;
    }

    // success
    return 0;
}

