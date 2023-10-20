#include <vector>

#ifndef ROOTINTERACTIVE_ROLLING_SUM
#define ROOTINTERACTIVE_ROLLING_SUM

using size_t = decltype(sizeof 1);

namespace RootInteractive {

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
OutputIt rolling_sum(InputIt first, InputIt last, OutputIt d_first, size_t kernel_width, T init, AddOp add, SubOp sub){
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
  for(;first+1 < last; ++first){
	  init = sub(std::move(init), *first);
	  *d_first = init;
	  ++d_first;
  }
  return d_first;
}

template <class InputIt, class OutputIt, class T>
OutputIt rolling_sum(InputIt first, InputIt last, OutputIt d_first, size_t radius, T init){
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
  for(;first+1 < last; ++first){
	  init = std::move(init) - *first;
	  *d_first = init;
	  ++d_first;
  }
  return d_first;
}

template <class InputIt, class OutputIt, class T>
OutputIt rolling_sum_symmetric(InputIt first, InputIt last, OutputIt d_first, size_t width, T init){
	// compute the differences
	OutputIt d_last = d_first;
	InputIt high = first+width;
	InputIt low = first;
	while(low<last){
		*d_last = *high - *low; 
		++d_last;
		++high;
		high = high == last ? first : high;
		++low;
	}	
	// now sum it up
	return std::exclusive_scan(d_first, d_last, d_first, std::reduce(first, first+width)+init);
}

template <class InputIt, class WeightsIt, class OutputIt, class T, class DistT>
OutputIt rolling_sum_weighted(InputIt first, InputIt last, WeightsIt w_first, OutputIt d_first, DistT radius, T init){
	InputIt low = first;
	InputIt high = first;
	WeightsIt w_low = w_first;
	WeightsIt w_high = w_first;
	for(;first<last;++first){
		while(high < last && *w_first + radius > *w_high){
			init = std::move(init) + *high;
			++high;
			++w_high;
		}
		while(*w_first - radius > *w_low){
			init = std::move(init) - *low;
			++low;
			++w_low;
		}
		*d_first = init;
		++d_first;
		++w_first;
	}
	return d_first;
}
}

#endif
