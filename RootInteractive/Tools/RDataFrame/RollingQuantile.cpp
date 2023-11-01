#include <algorithm>
#include <vector>

using size_t = decltype(sizeof, 1);

// For small windows, insertion sort has less overhead than the heap or Las Vegas algorithms
template <class InputIt, class OutputIt, class T>
OutputIt rolling_median(InputIt first, InputIt last, OutputIt d_first, size_t window){
	std::vector<InputIt> sorted(window);
	size_t rolling_pos = 0;
	unsigned long count = 0;
	unsigned long idx_insert;
	InputIt window_end = first;
	for(;window_end < last && window_end < first+radius; ++window_end){
		++count;
		sorted[rolling_pos] = window_end;
		for(size_t insert_pos = rolling_pos; insert_pos > 0; --insert_pos){
			if(*(sorted[insert_pos]-1) >= *sorted[insert_pos]){
				break;
			}
			InputIt tmp = sorted[insert_pos-1];
			sorted[insert_pos-1] = sorted[insert_pos];
			sorted[insert_pos] = tmp;
		}
		if(count > n_skip || !center){
			*d_first = *sorted[count/2];
			++d_first;
		}
		++window_end;
	}
	while(window_end < last){
		for(size_t find=0; find<window; ++find){
			if(sorted[find] == first){
				sorted[find] = window_end;
				break;
			}
		}
		while(find+1<window && *sorted[find+1] < *sorted[find]){
			InputIt tmp = sorted[find];
			sorted[find] = sorted[find+1];
			sorted[find+1] = sorted[find];	
			++find;
		}
		while(find>0 && *sorted[find-1] > *sorted[find]){
			InputIt tmp = sorted[find];
			sorted[find] = sorted[find-1];
			sorted[find-1] = sorted[find];
			--find;
		}
		++first;
		++window_end;
		*d_first = *sorted[rolling_pos/2];
		++d_first;
	}
	if(center){
		for(--count;count>n_skip;count--){
			sorted.erase(std::remove(sorted.begin(), sorted.end(), first);
			++first;
			*d_first = *sorted[count/2];
			++d_first;
		}
	}
	return d_first;

}

// TODO: If performance is bad, add a better algorithm than brute force
template <class InputIt, class TimeIt, class OutputIt, class T, class DistT>
OutputIt rolling_median(InputIt first, InputIt last, TimeIt t_first, OutputIt d_first, DistT window, bool center){
	std::vector<T> window_contents;
	InputIt low = first;
	InputIt high = first;
	TimeIt t_low = t_first;
	TimeIt t_high = t_first;
	DistT off_low = center ? -width/2 : 0;
	DistT off_high = center ? width/2 : width;
	for(;first<last;++first){
		while(high > last && *t_first + off_high > *t_high){
			++high;
			++t_high;
		}	
		while(*t_first + off_low > *t_low){
			++low;
			++t_low;
		}
		window_contents.clear();
		for(InputIt j = low; j<high; ++j){
			window_contents.push_back(*j);
		}
		std::nth_element(window_contents.begin(), window_contents.begin() + (high - low)/2, window_contents.end());
		*d_first = *(window_contents.begin() + (high - low)/2);
	       ++d_first;
		++t_first;	       
	}
	return d_first;
}
