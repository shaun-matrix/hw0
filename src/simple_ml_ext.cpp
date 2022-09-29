#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace std;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iters_per_epoch = m / batch;
    cout << "iters_per_epoch: " << iters_per_epoch << endl;
    float identity_matrix[batch][k];
    float z_matrix[batch][k];
    float gradient[n][k];

    for(int iter = 0; iter < iters_per_epoch; iter++){
        // locate X, y pointer in each batch
        const float *X_batch = X + iter*batch*n;
        const unsigned char *y_batch = y + iter*batch;

        for(size_t i = 0; i < batch; i++){
          // construct an identity matrix
          unsigned char label = y_batch[i];
          for(size_t j = 0; j < k; j++){
              if(j == label){
                identity_matrix[i][j] = 1;
              }
              else{
                identity_matrix[i][j] = 0;
              }
          }
        }

        for(size_t i = 0; i < batch; i++){
          // to normalize row-wise
          float sum_row = 0;
          // calculate exp(X * theta) and record sum row-wise
          for(size_t j = 0; j < k; j++){
            z_matrix[i][j] = 0;
            for(size_t l = 0; l < n; l++){
              z_matrix[i][j] += X_batch[i*n + l] * theta[l*k + j];
            }
            z_matrix[i][j] = exp(z_matrix[i][j]);
            sum_row += z_matrix[i][j];

          }
          // calculate Z = normalize(exp(X*theta)) and Z-I_y
          for(size_t j = 0; j < k; j++){
            z_matrix[i][j] = z_matrix[i][j] / sum_row - identity_matrix[i][j];
          }
        }

        // calculate gradient per batch
        for(size_t l=0; l<n; l++){
          for(size_t j=0; j<k; j++){
            gradient[l][j] = 0;
            for(size_t i=0; i<batch; i++){
              gradient[l][j] += X_batch[i*n+l] * z_matrix[i][j];
            }
            gradient[l][j] /= batch;
          }
        }

        // update theta; theta -= lr*gradient
        for(size_t l=0; l<n; l++){
          for(size_t j=0; j<k; j++){
            theta[l*k+j] -= lr * gradient[l][j];
          }
        } 

    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
