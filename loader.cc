#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/tensor_c_api.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <memory>
#include <stdio.h>

using namespace tensorflow;


void b(){std::cout<<"bb"<<std::endl;}

void sample(std::vector<std::vector<double>> &x, std::vector<double> &y, std::vector<std::vector<double>> &x_new, std::vector<int> &y_new, int batch_size, int num_steps)
{
  int N = x.size();
  //int ran  =  (int)(rand()*(N-batch_size+1.0)/(1.0+RAND_MAX));
  int ran = 0; //for comparison with Python API
  for(int i=0; i < batch_size; ++i){
    for(int j=0; j < num_steps; ++j){
      x_new[i].push_back(x[ran+i+(N/2)][j]);
    } 
  }

  for(int i=0; i < batch_size; ++i){
    y_new.push_back((int)y[ran+i+(N/2)]);
  }
}

void load_dataset(string filename, std::vector<std::vector<double>> &X_test_norm, std::vector<double> &y_test_norm){
  
  string str;
  int p;
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

  double v = 0.0;
  X_test_norm.resize(values.size());

  for(unsigned int i = 0; i < values.size(); ++i){
    for(unsigned int j = 0; j < values[i].size(); ++j){
      v = std::stod(values[i][j]);
      if(j != 0){
	X_test_norm[i].push_back(v);
      }
      else
	y_test_norm.push_back(v);
    }
  }
}

int abc(void){
  std::cout<<"bb"<<std::endl;
  int b_size = 15; // number of minibatch
  int n_step = 16; // size of minibatch

  std::vector<std::vector<double>> X_test_norm;
  std::vector<double> y_test_norm;

  load_dataset("/home/suzuki/LSTM_tsc-master/data/TEST_batch2000", X_test_norm, y_test_norm);

 std::cout<<"bb1"<<std::endl;
  // Initialize a tensorflow session
  Session* session;
  std::cout<<"bb1.5"<<std::endl;
  Status status = NewSession(SessionOptions(), &session);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
 std::cout<<"bb2"<<std::endl;
  // Read in the protobuf graph we exported
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
  std::cout<<"bb3"<<std::endl;
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
  c.scalar<float>()() = 1.0;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "input_data", a },
    { "Targets", b },
    { "Drop_out_keep_prob", c },
  };
  
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  //Run the session, evaluating our "costvalue, accuracy" operation from the graph
  status = session->Run(inputs, {"Softmax/costvalue","Softmax/accu","Softmax/Sparse_softmax/Sparse_softmax","Softmax_params/softmax_w"}, {}, &outputs);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // convert the node to a scalar representation.
  auto output_cost = outputs[0].scalar<float>();
  auto output_accuracy = outputs[1].scalar<float>();
  auto output_Sparse_softmax = outputs[2].tensor<float,1>();
  auto output_softmax_w = outputs[3].matrix<float>();

  float cost = output_cost(); 

  // (There are similar methods for vectors and matrices here:

  // Print the results
  for(unsigned int i = 0; i < output_softmax_w.size(); ++i){
    std::cout <<  "[" <<output_softmax_w(i,0) <<  "," <<output_softmax_w(i,1) <<"]"<< "\n";
  }

  std::cout <<"[";
  for(unsigned int i = 0; i < output_Sparse_softmax.size(); ++i){
    std::cout << output_Sparse_softmax(i) << " ";
  }
  std::cout <<"]"<<std::endl;

  std::cout << "output_cost() = " << std::setprecision(3) << cost/(b_size) << "\n"; 
  std::cout << "output_accuracy() = " << std::setprecision(2) << output_accuracy()  << "\n"; 

  // Free any resources used by the session
  session->Close();
  return 0;
}

