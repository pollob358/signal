import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, INF):
        self.x = np.zeros(INF, dtype=float)
        
    def set_value_at_time(self, t, value):
        # Set the value at time index t
        self.x[t] = value

    def shift(self, k):
      N = len(self.x)
    
      if k == 0:
        new = Signal(N)
        new.x = self.x.copy()
        return new

      if k > 0:
        new_len = N + k
        new = Signal(new_len)
        new.x[k:] = self.x
      else:
        new_len = N - k
        new = Signal(new_len)
        new.x[:N] = self.x[-k:]
      return new



    def add(self, other):
     # Determine the length of the result
      new_len = max(len(self.x), len(other.x))
      new = Signal(new_len)

     # Copy self.x into new.x
      new.x[:len(self.x)] = self.x

     # Add other.x, with zero-padding if needed
      new.x[:len(other.x)] += other.x

      return new

    def multiply(self, scalar):
        # Multiply a constant value with the signal
        new = Signal(len(self.x))
        new.x = scalar * self.x
        return new

    def plot(self, title="Discrete Signal"):
        # Plot the signal
        n = np.arange(len(self.x))  # fixed typo: arrange → arange
        plt.figure()
        plt.stem(n, self.x, use_line_collection=True)  # corrected stem usage
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel("x[n]")
        plt.grid(True)
        plt.show()


class LTI_System:
   def __init__(self, impulse_response: Signal):
        self.h = impulse_response  

   def linear_combination_of_impulses(self, input_signal: Signal):
       
        impulses=[]
        coeffs=[]
        N=len(input_signal.x)
        for k in range(N):
            if input_signal.x[k]!=0:
              delta=Signal(N)
              delta.set_value_at_time(k,1)
              impulses.append(delta)
              coeffs.append(input_signal.x[k])
        return impulses,coeffs    
        
   def output(self, input_signal: Signal):
        impulses, coeffs = self.linear_combination_of_impulses(input_signal)

        h_signal = Signal(len(self.h.x))
        h_signal.x = self.h.x.copy()

        N = len(input_signal.x)
        M = len(self.h.x)
        y_len = N + M - 1

        output_signal = Signal(y_len)

        for impulse, coeff in zip(impulses, coeffs):
            k = np.argmax(impulse.x)
            shifted_h = h_signal.shift(k)
            scaled_h = shifted_h.multiply(coeff)
            output_signal = output_signal.add(scaled_h)

        return output_signal


       
def read_signal_from_file(filename):
    with open(filename, "r") as f:
        nstart, nend = map(int, f.readline().split())
        values = list(map(float, f.readline().split()))

    N = nend - nstart + 1
    signal = Signal(N)

    for i, val in enumerate(values):
        signal.set_value_at_time(i, val)

    return signal, nstart, nend
               







if __name__ == "__main__":
    input_signal, nstart, nend = read_signal_from_file("input_signal.txt")

    n_axis = np.arange(nstart, nend + 1)
    plt.figure()
    plt.stem(n_axis, input_signal.x, use_line_collection=True)
    plt.title("Noisy Input Signal")
    plt.xlabel("n")
    plt.ylabel("x[n]")
    plt.grid(True)
    # plt.show()

    h = Signal(5)
    for i in range(5):
        h.set_value_at_time(i, 1/5)

    system = LTI_System(h)

    output_signal = system.output(input_signal)
    n_out = np.arange(nstart, nend + 5)
    plt.figure()
    plt.stem(n_out, output_signal.x, use_line_collection=True)
    plt.title("Smoothed Output Signal")
    plt.xlabel("n")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()
