#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

//#include "batch.cc"

using namespace tensorflow;

std::vector<std::vector<double>> sample_batch(std::vector<std::vector<double>> X_train , std::vector<double> y_train, int batch_size,int num_steps){
  int N = X_train.size();
  //int len = X_train[0].size();
  std::vector<std::vector<double>> X_batch(batch_size);
  int ran  =  (int)(rand()*(N-batch_size+1.0)/(1.0+RAND_MAX));
  int ind_N[batch_size];
  
  for(int i=0; i < batch_size; ++i){
    X_batch[i].push_back(X_train[ran+i][0]);
    for(int j=0; j < num_steps; ++j){
      X_batch[i].push_back(X_train[ran+i][j+1]);
    } 
  }

  return X_batch;

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

int main(int argc, char* argv[]) {

  std::vector<std::vector<string>> testvalues;                                                
  testvalues = load_dataset("data/TEST_batch1000");
  std::vector<std::vector<string>> trainvalues;                                               
  trainvalues = load_dataset("data/TEST_batch1000");
  std::stringstream ss;
  double v = 0.0;
  std::string st;
  int linenum = 1000;

  std::vector<std::vector<double>> X_train_norm(linenum), X_test_norm(linenum);
  std::vector<double> y_train_norm(linenum), y_test_norm(linenum);


  for(unsigned int i = 0; i < trainvalues.size(); ++i){
    for(unsigned int j = 0; j < trainvalues[i].size(); ++j){
      v = std::stod(trainvalues[i][j]);
      if(j != 0)
	X_train_norm[i].push_back(v);
      else
	y_train_norm.push_back(v);
    }
  }

  for(unsigned int i = 0; i < testvalues.size(); ++i){
    for(unsigned int j = 0; j < testvalues[i].size(); ++j){
      v = std::stod(trainvalues[i][j]);
      if(j != 0){
	X_test_norm[i].push_back(v);
      }
      else
	y_test_norm.push_back(v);
    }
  }

  std::vector<std::vector<double>> batchx = sample_batch(X_test_norm, y_test_norm, 16, 15);

  for(unsigned int i = 0; i < batchx.size(); ++i){
    for(unsigned int j = 0; j < batchx[i].size(); ++j){
      std::cout<< batchx[i][j] <<std::endl;
    }
    std::cout << i << std::endl;
  }

  std::cout<< batchx.size() <<std::endl;
  std::cout<< batchx[0].size() <<std::endl;


  Eigen::Vector3d A = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(&(batchx[0][0]), 16,16);

  std::cout << A << std::endl;


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

  Tensor a(DT_FLOAT, TensorShape({15,16}));
  //a.scalar<float>()() = 3.0;
  a.flat_inner_dims<float>().setRandom();
  //a.flat_inner_dims<float>() = batchx;
  //std::cout<<"a.matrix<T>() is "<< a.shaped<double, 2>({4, 15})<<std::endl;
  std::cout<<"a.flat_inner_dims<float>() is "<< a.flat_inner_dims<float>() <<std::endl;
  //std::cout<<"a.vec<T>() is "<< a.vec<float>()<<std::endl;

  Tensor b(DT_INT64, TensorShape({15}));
  b.flat<int64>().setRandom();
  //std::cout<<"b.matrix<double>() is "<< b.matrix<double>()<<std::endl;

  Tensor c(DT_FLOAT, TensorShape());
  c.scalar<float>()() = 0.5;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "input_data", a },
    { "Targets", b },
    { "Drop_out_keep_prob", c },
  };
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  //Run the session, evaluating our "c" operation from the graph
    status = session->Run(inputs, {"Softmax/costvalue"}, {}, &outputs);
  //status = session->Run(input_data: X_batch,targets:y_batch, initial_state:state,keep_prob:1});

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}
