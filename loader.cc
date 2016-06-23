#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iterator>
#include <memory>

//#include "batch.cc"

#include <stdio.h>

using namespace tensorflow;

int b_size = 15;
int n_step = 16;

void sample(std::vector<std::vector<double>> &x, std::vector<double> &y, std::vector<std::vector<double>> &x_new, std::vector<int> &y_new, int batch_size, int num_steps);

void sample(std::vector<std::vector<double>> &x, std::vector<double> &y, std::vector<std::vector<double>> &x_new, std::vector<int> &y_new, int batch_size, int num_steps)
{
  int N = x.size();
  int ran  =  (int)(rand()*(N-batch_size+1.0)/(1.0+RAND_MAX));
  for(int i=0; i < batch_size; ++i){
    for(int j=0; j < num_steps; ++j){
      x_new[i].push_back(x[ran+i][j]);
    } 
  }

  for(int i=0; i < batch_size; ++i){
    y_new.push_back((int)y[ran+i]);
  }

}

std::vector<std::vector<string>> load_dataset(string filename){
  string str;
  int p;
  std::stringstream ss;                                                                                                    
  std::vector<std::vector<string>> values;
  std::ifstream file(filename);

  while(getline(file, str)){
    std::vector<string> inner;
    while( (p = str.find(",")) != string::npos ){                                                                              
      inner.push_back(str.substr(0, p));  
      str = str.substr(p+1);                                                                                                       
    }
    inner.push_back(str);                                                                       
    values.push_back(inner);
  }                           

  return values;
}

int main(void) {

  std::vector<std::vector<string>> testvalues;                                                
  testvalues = load_dataset("data/TEST_batch1000");
  std::stringstream ss;
  double v = 0.0;
  std::string st;
  int linenum = 1000;

  std::vector<std::vector<double>> X_test_norm(linenum);
  std::vector<double> y_test_norm(linenum);

  for(unsigned int i = 0; i < testvalues.size(); ++i){
    for(unsigned int j = 0; j < testvalues[i].size(); ++j){
      v = std::stod(testvalues[i][j]);
      if(j != 0){
	X_test_norm[i].push_back(v);
      }
      else
	y_test_norm.push_back(v);
    }
  }


  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;

  status = ReadBinaryProto(Env::Default(), "/home/suzuki/LSTM_tsc-master/models/output_graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Setup inputs and outputs:
  std::vector<std::vector<double>> batchx(b_size);
  std::vector<int> batchy;
  
  sample(X_test_norm, y_test_norm, batchx, batchy, b_size, n_step);

  Tensor a(DT_FLOAT, TensorShape({b_size,n_step}));
  for(unsigned int i = 0; i < batchx.size(); ++i){
    for(unsigned int j = 0; j < batchx[i].size(); ++j){
      a.matrix<float>()(i,j) = batchx[i][j];
    }
  }

  Tensor b(DT_INT64, TensorShape({b_size}));
  b.flat<int64>().setZero();
  for(unsigned int i = 0; i < batchy.size(); ++i){
    b.tensor<int64,1>()(i) = batchy[i];
  }

  Tensor c(DT_FLOAT, TensorShape());
  c.scalar<float>()() = 0.5;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "input_data", a },
    { "Targets", b },
    { "Drop_out_keep_prob", c },
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  //Run the session, evaluating our "costvalue, accuracy" operation from the graph
  status = session->Run(inputs, {"Softmax/costvalue","Softmax/accu"}, {}, &outputs);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // convert the node to a scalar representation.
  auto output_cost = outputs[0].scalar<float>();
  auto output_accuracy = outputs[1].scalar<float>();

  // (There are similar methods for vectors and matrices here:

  // Print the results
  std::cout << output_cost() << "\n"; // 30
  std::cout << output_accuracy() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}

