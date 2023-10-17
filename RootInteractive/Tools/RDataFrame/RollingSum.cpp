#include <vector>

#ifndef ROOTINTERACTIVE_ROLLING_SUM
#define ROOTINTERACTIVE_ROLLING_SUM

using size_t = decltype(sizeof 1);

template <class InputIt, class T>
std::vector<T> rolling_sum(InputIt first, InputIt last, size_t radius, T init){
	std::vector<T> result(last-first);
	InputIt window_end = first;

	return result;
}

  //size_t arr_size = radius < last - first ? last-first+radius : 2*(last-first)-1;
  //std::vector<T> result(arr_size);
  //
  //TODO: Find out if it's automatically parallelized - should be, as it should find the "scan" operation
  //Algorithm works single threaded if operation is associative and has a left inverse, but parallel only works if op is also commutative and has an inverse
template <class InputIt, class OutputIt, class T, class AddOp, class SubOp>
void rolling_sum(InputIt first, InputIt last, OutputIt d_first, size_t kernel_width, T init, AddOp add, SubOp sub){
  InputIt window_end = first;
  for(;window_end < last && window_end < first + kernel_width; ++window_end){
	  init = add(std::move(init), *window_end);
	  *d_first = init;
	  ++d_first;
  }
  while(window_end < last){
	  init = add(std::move(init), *window_end);
	  init = sub(std::move(init), *first);
	  ++first;
	  ++window_end;
	  *d_first = init;
	  ++d_first;
  }
  for(;first < last; ++first){
	  init = sub(std::move(init), *first);
	  *d_first = init;
	  ++d_first;
  }
}

template <class InputIt, class OutputIt, class T>
void rolling_sum(InputIt first, InputIt last, OutputIt d_first, size_t radius, T init){
  InputIt window_end = first;
  for(;window_end < last && window_end < first + radius; ++window_end){
	  init = std::move(init) + *window_end;
	  *d_first = init;
	  ++d_first;
  }
  while(window_end < last){
	  init = std::move(init) + *window_end;
	  init = std::move(init) - *first;
	  ++first;
	  ++window_end;
	  *d_first = init;
	  ++d_first;
  }
  for(;first < last; ++first){
	  init = std::move(init) - *first;
	  *d_first = init;
	  ++d_first;
  }
}

#endif
