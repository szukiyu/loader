#ifndef INCLUDED_LOADER_H
#define INCLUDED_LOADER_H

#include <stdio.h>
#include <vector>
#include <string>

int abc(void);
void b();

void sample(std::vector<std::vector<double> > &x, std::vector<double> &y, std::vector<std::vector<double> > &x_new, std::vector<int> &y_new, int batch_size, int num_steps);

void load_dataset(std::string filename, std::vector<std::vector<double> > &X_test_norm, std::vector<double> &y_test_norm);

int main(void);

#endif/*INCLUDED_LOADER_H*/
