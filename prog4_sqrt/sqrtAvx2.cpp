#include <cmath>
#include <cstdio>
#include <immintrin.h>
#include <omp.h>

// 使用 AVX 指令集实现平方根计算（牛顿迭代法）
// N: 数组长度
// initialGuess: 迭代初始值
// values: 输入数组，存储需要计算平方根的数
// output: 输出数组，存储计算结果
// 方程：1 / guess^2 = x
// !!!: x 不是迭代变量，guess 才是，且这里的 x 是数组的元素
// !!!: sqrt(x) = 1 / guess = x * guess
void sqrt_avx(int N, float initialGuess, float *values, float *output) {
    // 输入参数有效性检查
    if (!values || !output || N <= 0) return;

    // 设置收敛阈值
    static const float kThreshold = 0.00001f;
    
    // 计算能被 8 整除的最大长度（AVX2 可以同时处理 8 个 float）
    int aligned_n = (N / 8) * 8;

    // 使用 AVX 指令并行处理每 8 个元素
    for (int i = 0; i < aligned_n; i += 8) {
        // 将阈值广播到 256 位向量中（8 个 float）
        __m256 threshold = _mm256_set1_ps(kThreshold);
        // 从内存加载 8 个连续的 float 到向量寄存器
        __m256 x_vec = _mm256_loadu_ps(&values[i]);
        // 将初始猜测值广播到向量寄存器
        __m256 guess = _mm256_set1_ps(initialGuess);

        // 计算初始误差：|guess² * x - 1|
        __m256 pred = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_mul_ps(guess, guess), x_vec),
            _mm256_set1_ps(1.0f)
        );
        // 计算误差的绝对值
        pred = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), pred);
        
        /*
        1. 在浮点数的二进制表示中，最高位（符号位）为：
            - 0：表示正数
            - 1：表示负数

        2. `-0.0f` 的二进制表示是 `1000 0000 0000 0000 0000 0000 0000 0000`
        - 即只有符号位为 1，指数和尾数位都是 0

        3. `_mm256_andnot_ps(a, b)` 的操作相当于 `~a & b`，即：
        - 先对第一个操作数取位反（NOT）
        - 然后与第二个操作数进行位与（AND）

        4. 所以 `_mm256_andnot_ps(_mm256_set1_ps(-0.0f), x)` 的过程是：
        ```
        x:              [sign][exponent][mantissa] (原始数)
        -0.0f:          1000...000      (全 0，仅符号位为 1)
        ~(-0.0f):       0111...111      (全 1，仅符号位为 0)
        ~(-0.0f) & x:   0[exponent][mantissa]     (保持数值位不变，符号位强制为 0)
        ```

        这样就巧妙地将任何负数转换为对应的正数，而正数保持不变，从而实现了绝对值的计算。
        */  


        // 当任意一个元素的误差大于阈值时继续迭代
        while (_mm256_movemask_ps(_mm256_cmp_ps(pred, threshold, _CMP_GT_OQ)) != 0) {
            // _mm256_movemask_ps 是 AVX 指令集中的一个函数，用于从 256 位浮点向量中提取每个元素的符号位，并将这些符号位组合成一个 8 位的整数。
            // 牛顿迭代公式：guess = (3 * guess - x * guess³) * 0.5
            guess = _mm256_mul_ps(
                _mm256_set1_ps(0.5f),
                _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_set1_ps(3.0f), guess),
                    _mm256_mul_ps(x_vec, _mm256_mul_ps(guess, _mm256_mul_ps(guess, guess)))
                )
            );
            
            // 重新计算误差
            pred = _mm256_sub_ps(
                _mm256_mul_ps(_mm256_mul_ps(guess, guess), x_vec),
                _mm256_set1_ps(1.0f)
            );
            pred = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), pred);
        }

        // 计算最终结果 sqrt(x) = x * guess，并存储到输出数组
        _mm256_storeu_ps(&output[i], _mm256_mul_ps(x_vec, guess));
    }

    // 使用标准算法处理剩余的元素（不足 8 个的部分）
    for (int i = aligned_n; i < N; i++) {
        float x = values[i];
        float guess = initialGuess;
        float error = fabs(guess * guess * x - 1.f);
        
        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }
        
        output[i] = x * guess;
    }
}

// 使用 AVX 指令集实现平方根计算（牛顿迭代法）
// 使用掩码，当满足阈值时不继续迭代 (这个案例其实没有必要)
// N: 数组长度
// initialGuess: 迭代初始值
// values: 输入数组，存储需要计算平方根的数
// output: 输出数组，存储计算结果
// 方程：1 / guess^2 = x
// !!!: x 不是迭代变量，guess 才是，且这里的 x 是数组的元素
// !!!: sqrt(x) = 1 / guess = x * guess
void sqrt_avx_mask(int N, float initialGuess, float *values, float *output) {
    // 输入参数有效性检查
    if (!values || !output || N <= 0) return;

    // 设置收敛阈值
    static const float kThreshold = 0.00001f;
    
    // 计算能被 8 整除的最大长度（AVX2 可以同时处理 8 个 float）
    int aligned_n = (N / 8) * 8;

    // 使用 AVX 指令并行处理每 8 个元素
    for (int i = 0; i < aligned_n; i += 8) {
        // 将阈值广播到 256 位向量中（8 个 float）
        __m256 threshold = _mm256_set1_ps(kThreshold);
        // 从内存加载 8 个连续的 float 到向量寄存器
        __m256 x_vec = _mm256_loadu_ps(&values[i]);
        // 将初始猜测值广播到向量寄存器
        __m256 guess = _mm256_set1_ps(initialGuess);

        // 计算初始误差：|guess² * x - 1|
        __m256 pred = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_mul_ps(guess, guess), x_vec),
            _mm256_set1_ps(1.0f)
        );
        // 计算误差的绝对值
        pred = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), pred);
        
        /*
        1. 在浮点数的二进制表示中，最高位（符号位）为：
            - 0：表示正数
            - 1：表示负数

        2. `-0.0f` 的二进制表示是 `1000 0000 0000 0000 0000 0000 0000 0000`
        - 即只有符号位为 1，指数和尾数位都是 0

        3. `_mm256_andnot_ps(a, b)` 的操作相当于 `~a & b`，即：
        - 先对第一个操作数取位反（NOT）
        - 然后与第二个操作数进行位与（AND）

        4. 所以 `_mm256_andnot_ps(_mm256_set1_ps(-0.0f), x)` 的过程是：
        ```
        x:              [sign][exponent][mantissa] (原始数)
        -0.0f:          1000...000      (全 0，仅符号位为 1)
        ~(-0.0f):       0111...111      (全 1，仅符号位为 0)
        ~(-0.0f) & x:   0[exponent][mantissa]     (保持数值位不变，符号位强制为 0)
        ```

        这样就巧妙地将任何负数转换为对应的正数，而正数保持不变，从而实现了绝对值的计算。
        */  

        // 初始化掩码为全 1
        __m256 mask = _mm256_cmp_ps(pred, threshold, _CMP_GT_OQ);
        /*
        常用的比较操作符（imm8 的值）：
            _CMP_EQ_OQ：相等
            _CMP_LT_OQ：小于
            _CMP_LE_OQ：小于或等于
            _CMP_GT_OQ：大于
            _CMP_GE_OQ：大于或等于
            _CMP_NEQ_OQ：不等于
        */

        // 当任意一个元素的误差大于阈值时继续迭代
        while (_mm256_movemask_ps(mask) != 0) {
            // _mm256_movemask_ps 是 AVX 指令集中的一个函数，用于从 256 位浮点向量中提取每个元素的符号位，并将这些符号位组合成一个 8 位的整数。
            // 牛顿迭代公式：guess = (3 * guess - x * guess³) * 0.5
            __m256 new_guess = _mm256_mul_ps(
                _mm256_set1_ps(0.5f),
                _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_set1_ps(3.0f), guess),
                    _mm256_mul_ps(x_vec, _mm256_mul_ps(guess, _mm256_mul_ps(guess, guess)))
                )
            );

            // 仅更新掩码为 1 的元素
            guess = _mm256_blendv_ps(guess, new_guess, mask);
            /*
            _mm256_blendv_ps 用于根据掩码在两个 256 位浮点向量之间进行条件选择。它允许在两个向量中选择性地从一个向量复制元素到结果向量中，具体选择由掩码决定。
                参数：
                a：第一个 256 位浮点向量。
                b：第二个 256 位浮点向量。
                mask：一个 256 位浮点向量，用于控制选择。
                返回值：
                返回一个新的 256 位浮点向量。对于每个元素，如果 mask 中对应位置的符号位为 1，则从 b 中选择该元素；否则，从 a 中选择该元素。
            */

            // 重新计算误差
            pred = _mm256_sub_ps(
                _mm256_mul_ps(_mm256_mul_ps(guess, guess), x_vec),
                _mm256_set1_ps(1.0f)
            );
            pred = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), pred);

            // 更新掩码
            mask = _mm256_cmp_ps(pred, threshold, _CMP_GT_OQ);
        }

        // 计算最终结果 sqrt(x) = x * guess，并存储到输出数组
        _mm256_storeu_ps(&output[i], _mm256_mul_ps(x_vec, guess));
    }

    // 使用标准算法处理剩余的元素（不足 8 个的部分）
    for (int i = aligned_n; i < N; i++) {
        float x = values[i];
        float guess = initialGuess;
        float error = fabs(guess * guess * x - 1.f);
        
        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }
        
        output[i] = x * guess;
    }
}
