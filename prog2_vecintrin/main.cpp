#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <math.h>
#include "CS149intrin.h"
#include "logger.h"
using namespace std;

#define EXP_MAX 10

Logger CS149Logger;

void usage(const char* progname);
void initValue(float* values, int* exponents, float* output, float* gold, unsigned int N);
void absSerial(float* values, float* output, int N);
void absVector(float* values, float* output, int N);
void clampedExpSerial(float* values, int* exponents, float* output, int N);
void clampedExpVector(float* values, int* exponents, float* output, int N);
float arraySumSerial(float* values, int N);
float arraySumVector(float* values, int N);
bool verifyResult(float* values, int* exponents, float* output, float* gold, int N);

int main(int argc, char * argv[]) {
  int N = 16;
  bool printLog = false;

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
    {"size", 1, 0, 's'},
    {"log", 0, 0, 'l'},
    {"help", 0, 0, '?'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "s:l?", long_options, NULL)) != EOF) {

    switch (opt) {
      case 's':
        N = atoi(optarg);
        if (N <= 0) {
          printf("Error: Workload size is set to %d (<0).\n", N);
          return -1;
        }
        break;
      case 'l':
        printLog = true;
        break;
      case '?':
      default:
        usage(argv[0]);
        return 1;
    }
  }


  float* values = new float[N+VECTOR_WIDTH];
  int* exponents = new int[N+VECTOR_WIDTH];
  float* output = new float[N+VECTOR_WIDTH];
  float* gold = new float[N+VECTOR_WIDTH];
  initValue(values, exponents, output, gold, N);

  clampedExpSerial(values, exponents, gold, N);
  clampedExpVector(values, exponents, output, N);

  //absSerial(values, gold, N);
  //absVector(values, output, N);

  printf("\e[1;31mCLAMPED EXPONENT\e[0m (required) \n");
  bool clampedCorrect = verifyResult(values, exponents, output, gold, N);
  if (printLog) CS149Logger.printLog();
  CS149Logger.printStats();

  printf("************************ Result Verification *************************\n");
  if (!clampedCorrect) {
    printf("@@@ Failed!!!\n");
  } else {
    printf("Passed!!!\n");
  }

  printf("\n\e[1;31mARRAY SUM\e[0m (bonus) \n");
  if (N % VECTOR_WIDTH == 0) {
    float sumGold = arraySumSerial(values, N);
    float sumOutput = arraySumVector(values, N);
    float epsilon = 0.1;
    bool sumCorrect = abs(sumGold - sumOutput) < epsilon * 2;
    if (!sumCorrect) {
      printf("Expected %f, got %f\n.", sumGold, sumOutput);
      printf("@@@ Failed!!!\n");
    } else {
      printf("Passed!!!\n");
    }
  } else {
    printf("Must have N %% VECTOR_WIDTH == 0 for this problem (VECTOR_WIDTH is %d)\n", VECTOR_WIDTH);
  }

  delete [] values;
  delete [] exponents;
  delete [] output;
  delete [] gold;

  return 0;
}

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -s  --size <N>     Use workload size N (Default = 16)\n");
  printf("  -l  --log          Print vector unit execution log\n");
  printf("  -?  --help         This message\n");
}

void initValue(float* values, int* exponents, float* output, float* gold, unsigned int N) {

  for (unsigned int i=0; i<N+VECTOR_WIDTH; i++)
  {
    // random input values
    values[i] = -1.f + 4.f * static_cast<float>(rand()) / RAND_MAX;
    exponents[i] = rand() % EXP_MAX;
    output[i] = 0.f;
    gold[i] = 0.f;
  }

}

bool verifyResult(float* values, int* exponents, float* output, float* gold, int N) {
  int incorrect = -1;
  float epsilon = 0.00001;
  for (int i=0; i<N+VECTOR_WIDTH; i++) {
    if ( abs(output[i] - gold[i]) > epsilon ) {
      incorrect = i;
      break;
    }
  }

  if (incorrect != -1) {
    if (incorrect >= N)
      printf("You have written to out of bound value!\n");
    printf("Wrong calculation at value[%d]!\n", incorrect);
    printf("value  = ");
    for (int i=0; i<N; i++) {
      printf("% f ", values[i]);
    } printf("\n");

    printf("exp    = ");
    for (int i=0; i<N; i++) {
      printf("% 9d ", exponents[i]);
    } printf("\n");

    printf("output = ");
    for (int i=0; i<N; i++) {
      printf("% f ", output[i]);
    } printf("\n");

    printf("gold   = ");
    for (int i=0; i<N; i++) {
      printf("% f ", gold[i]);
    } printf("\n");
    return false;
  }
  printf("Results matched with answer!\n");
  return true;
}

// computes the absolute value of all elements in the input array
// values, stores result in output
void absSerial(float* values, float* output, int N) {
  for (int i=0; i<N; i++) {
    float x = values[i];
    if (x < 0) {
      output[i] = -x;
    } else {
      output[i] = x;
    }
  }
}


// implementation of absSerial() above, but it is vectorized using CS149 intrinsics
void absVector(float* values, float* output, int N) {
  __cs149_vec_float x;
  __cs149_vec_float result;
  __cs149_vec_float zero = _cs149_vset_float(0.f);
  __cs149_mask maskAll, maskIsNegative, maskIsNotNegative;

//  Note: Take a careful look at this loop indexing.  This example
//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
//  Why is that the case?
  for (int i=0; i<N; i+=VECTOR_WIDTH) {

    // All ones
    maskAll = _cs149_init_ones();

    // All zeros
    maskIsNegative = _cs149_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _cs149_vload_float(x, values+i, maskAll);               // x = values[i];

    // Set mask according to predicate
    _cs149_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _cs149_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _cs149_mask_not(maskIsNegative);     // } else {

    // Execute instruction ("else" clause)
    _cs149_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _cs149_vstore_float(output+i, result, maskAll);
  }
}


// accepts an array of values and an array of exponents
//
// For each element, compute values[i]^exponents[i] and clamp value to
// 9.999.  Store result in output.
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
  for (int i=0; i<N; i++) {
    float x = values[i];
    int y = exponents[i];
    if (y == 0) {
      output[i] = 1.f;
    } else {
      float result = x;
      int count = y - 1;
      while (count > 0) {
        result *= x;
        count--;
      }
      if (result > 9.999999f) {
        result = 9.999999f;
      }
      output[i] = result;
    }
  }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __cs149_vec_int zero_i = _cs149_vset_int(0);
  __cs149_vec_int one_i = _cs149_vset_int(1);
  __cs149_vec_float nine_999 = _cs149_vset_float(9.999999f);
  __cs149_mask mask_one = _cs149_init_ones(VECTOR_WIDTH); // 1 的 mask

  __cs149_vec_float s_x;
  __cs149_vec_int s_e;
  __cs149_vec_float results;
  
  __cs149_mask mask_lt_N; // 索引小于 N 的 mask
  __cs149_mask mask_unfinished; // 计算还没有结束的 mask
  __cs149_mask mask_zero_exp; // 指数为 0 的 mask
  __cs149_mask greater_than_9_999; // 大于 9.999 的 mask
  
  for (int i=0; i<N; i+=VECTOR_WIDTH) {
    mask_lt_N = _cs149_init_ones(min(N-i, VECTOR_WIDTH)); // 标记索引小于 N 的元素

    _cs149_vload_float(s_x, values+i, mask_lt_N); // 从 values[i] 开始，加载 VECTOR_WIDTH 个元素到 s_x
    _cs149_vload_int(s_e, exponents+i, mask_lt_N); // 从 exponents[i] 开始，加载 VECTOR_WIDTH 个元素到 s_e
    _cs149_vset_float(results, 1.f, mask_lt_N); // 将结果初始化为 1

    mask_unfinished = mask_lt_N; // 初始情况时，只要索引有效，则表示计算未结束
    greater_than_9_999 = _cs149_init_ones(0); // 初始化为 0

    _cs149_veq_int(mask_zero_exp, s_e, zero_i, mask_one); // 如果 exponent == 0, 则标记为 1 表示计算结束，否则标记为 0
    mask_zero_exp = _cs149_mask_not(mask_zero_exp); // 反转，如果 exp 为 0, 则表示计算结束，用 0 表示
    mask_unfinished = _cs149_mask_and(mask_unfinished, mask_zero_exp); // 如果指数为 0, 则表示计算结束
    mask_unfinished = _cs149_mask_and(mask_unfinished, mask_lt_N); // 保险起见，再和 mask_lt_N 取交集，保证索引无效的部分为 0

    while (_cs149_cntbits(mask_unfinished) > 0) {
      _cs149_vmult_float(results, results, s_x, mask_unfinished); // 计算一次乘方
      _cs149_vsub_int(s_e, s_e, one_i, mask_unfinished); // 指数自减

      _cs149_vgt_float(greater_than_9_999, results, nine_999, mask_unfinished); // 如果结果大于 9.999, 则标记为 1
      _cs149_vset_float(results, 9.999999f, greater_than_9_999); // 如果结果大于 9.999, 则将结果设置为 9.999

      // 更新 mask_unfinished
      mask_zero_exp = _cs149_init_ones(0); // 重新初始化 mask_zero_exp
      _cs149_veq_int(mask_zero_exp, s_e, zero_i, mask_one); // 如果 exponent == 0, 则标记为 1 表示计算结束，否则标记为 0

      __cs149_mask new_finished = _cs149_mask_or(mask_zero_exp, greater_than_9_999); // 如果指数为 0 或结果大于 9.999, 则表示计算结束，用 1 表示

      mask_zero_exp = _cs149_mask_not(new_finished); // 反转，计算结束的位置用 0 表示，计算未结束的位置用 1 表示
      mask_unfinished = _cs149_mask_and(mask_unfinished, mask_zero_exp); // 如果指数为 0, 则表示计算结束
      mask_unfinished = _cs149_mask_and(mask_unfinished, mask_lt_N); // 保险起见，再和 mask_lt_N 取交集，保证索引无效的部分为 0
    }

    _cs149_vstore_float(output+i, results, mask_lt_N); // 将结果存储到 output 中

    addUserLog("clampedExpVector");
  }
}

// returns the sum of all elements in values
float arraySumSerial(float* values, int N) {
  float sum = 0;
  for (int i=0; i<N; i++) {
    sum += values[i];
  }

  return sum;
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
  
  //
  // CS149 STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __cs149_vec_float v_sum;
  __cs149_mask mask_one = _cs149_init_ones(VECTOR_WIDTH); // 1 的 mask
  _cs149_vset_float(v_sum, 0.f, mask_one);

  __cs149_vec_float v_elems;
  
  for (int i=0; i<N; i+=VECTOR_WIDTH) {
    _cs149_vload_float(v_elems, values+i, mask_one); // 从 values[i] 开始，加载 VECTOR_WIDTH 个元素到 v_elems
    _cs149_vadd_float(v_sum, v_sum, v_elems, mask_one); // 将 v_elems 加到 v_sum 上
  }

  float sum = 0.f;
  for (int i=0; i<VECTOR_WIDTH; i++) {
    sum += v_sum.value[i];
  }

  return sum;
}

