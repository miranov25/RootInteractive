#ifndef ROOTINTERACTIVE_ROLLING_QUANTILE
#define ROOTINTERACTIVE_ROLLING_QUANTILE

#include <algorithm>
#include <vector>

namespace RootInteractive{

using size_t = decltype(sizeof 1);

// For small windows, insertion sort has less overhead than the heap or Las Vegas algorithms
template <class InputIt, class OutputIt>
OutputIt rolling_median(InputIt first, InputIt last, OutputIt d_first, size_t window, bool center){
	std::vector<InputIt> sorted;
	size_t rolling_pos = 0;
	unsigned long count = 0;
	unsigned long idx_insert;
	size_t n_skip = window >> 1;
	InputIt window_end = first;
	for(;window_end < last && window_end < first+window; ++window_end){
		++count;
		sorted.push_back(window_end);
		for(auto insert_pos = sorted.end()-1; insert_pos > sorted.begin(); --insert_pos){
			if(*(insert_pos-1) <= *insert_pos){
				break;
			}
			InputIt tmp = *insert_pos;
			*insert_pos = insert_pos[-1];
			insert_pos[-1] = tmp;
		}
		if(count > n_skip || !center){
			*d_first = *sorted[count/2];
			++d_first;
		}
		++rolling_pos;
	}
	while(window_end < last){
		size_t found;
		for(found=0; found<window; ++found){
			if(sorted[found] == first){
				sorted[found] = window_end;
				break;
			}
		}
		while(found+1<window && *sorted[found+1] < *sorted[found]){
			InputIt tmp = sorted[found];
			sorted[found] = sorted[found+1];
			sorted[found+1] = tmp;	
			++found;
		}
		while(found>0 && *sorted[found-1] > *sorted[found]){
			InputIt tmp = sorted[found];
			sorted[found] = sorted[found-1];
			sorted[found-1] = tmp;
			--found;
		}
		++first;
		++window_end;
		*d_first = *sorted[count/2];
		++d_first;
	}
	if(center){
		for(--count;count>n_skip;count--){
			sorted.erase(std::remove(sorted.begin(), sorted.end(), first));
			++first;
			*d_first = *sorted[count/2];
			++d_first;
		}
	}
	return d_first;

}

// TODO: If performance is bad, add a better algorithm than brute force
template <class InputIt, class TimeIt, class OutputIt, class DistT>
OutputIt rolling_median_weighted(InputIt first, InputIt last, TimeIt t_first, OutputIt d_first, DistT window, bool center){
	std::vector<InputIt> window_contents;
	InputIt low = first;
	InputIt high = first;
	TimeIt t_low = t_first;
	TimeIt t_high = t_first;
	DistT off_low = center ? -window/2 : 0;
	DistT off_high = center ? window/2 : window;
	for(;first<last;++first){
		while(high < last && *t_first + off_high > *t_high){
			++high;
			++t_high;
		}	
		while(*t_first + off_low > *t_low){
			++low;
			++t_low;
		}
		window_contents.clear();
		for(InputIt j = low; j<high; ++j){
			window_contents.push_back(j);
		}
		std::nth_element(window_contents.begin(), window_contents.begin() + (high - low)/2, window_contents.end(), [](InputIt a, InputIt b){return *a < *b;});
		*d_first = **(window_contents.begin() + (high - low)/2);
	       ++d_first;
		++t_first;	       
	}
	return d_first;
}

// For small windows, insertion sort has less overhead than the heap or Las Vegas algorithms
template <class InputIt, class OutputIt>
OutputIt rolling_quantile(InputIt first, InputIt last, OutputIt d_first, size_t window, double quantile, bool center){
	std::vector<InputIt> sorted;
	size_t rolling_pos = 0;
	unsigned long count = 0;
	unsigned long idx_insert;
	size_t n_skip = window >> 1;
	// Sanity checks on quantile to prevent buffer overflow
	if(quantile >= 1.){
		quantile = 1.;
	}
	if(quantile <= 0. || quantile != quantile){
		quantile = 0.;
	}
	InputIt window_end = first;
	for(;window_end < last && window_end < first+window; ++window_end){
		++count;
		sorted.push_back(window_end);
		for(auto insert_pos = sorted.end()-1; insert_pos > sorted.begin(); --insert_pos){
			if(*(insert_pos-1) <= *insert_pos){
				break;
			}
			InputIt tmp = *insert_pos;
			*insert_pos = insert_pos[-1];
			insert_pos[-1] = tmp;
		}
		if(count > n_skip || !center){
			*d_first = *sorted[(size_t)(count*quantile)];
			++d_first;
		}
		++rolling_pos;
	}
	while(window_end < last){
		size_t found;
		for(found=0; found<window; ++found){
			if(sorted[found] == first){
				sorted[found] = window_end;
				break;
			}
		}
		while(found+1<window && *sorted[found+1] < *sorted[found]){
			InputIt tmp = sorted[found];
			sorted[found] = sorted[found+1];
			sorted[found+1] = tmp;	
			++found;
		}
		while(found>0 && *sorted[found-1] > *sorted[found]){
			InputIt tmp = sorted[found];
			sorted[found] = sorted[found-1];
			sorted[found-1] = tmp;
			--found;
		}
		++first;
		++window_end;
		*d_first = *sorted[(size_t)(count*quantile)];
		++d_first;
	}
	if(center){
		for(--count;count>n_skip;count--){
			sorted.erase(std::remove(sorted.begin(), sorted.end(), first));
			++first;
			*d_first = *sorted[(size_t)(count*quantile)];
			++d_first;
		}
	}
	return d_first;

}

// TODO: If performance is bad, add a better algorithm than brute force
template <class InputIt, class TimeIt, class OutputIt, class DistT>
OutputIt rolling_quantile_weighted(InputIt first, InputIt last, TimeIt t_first, OutputIt d_first, DistT window, double quantile, bool center){
	std::vector<InputIt> window_contents;
	InputIt low = first;
	InputIt high = first;
	TimeIt t_low = t_first;
	TimeIt t_high = t_first;
	DistT off_low = center ? -window/2 : 0;
	DistT off_high = center ? window/2 : window;
	// Sanity checks on quantile to prevent buffer overflow
	if(quantile >= 1.){
		quantile = 1.;
	}
	if(quantile <= 0. || quantile != quantile){
		quantile = 0.;
	}
	for(;first<last;++first){
		while(high < last && *t_first + off_high > *t_high){
			++high;
			++t_high;
		}	
		while(*t_first + off_low > *t_low){
			++low;
			++t_low;
		}
		window_contents.clear();
		for(InputIt j = low; j<high; ++j){
			window_contents.push_back(j);
		}
		std::nth_element(window_contents.begin(), window_contents.begin() + (size_t)((high - low)*quantile), window_contents.end(), [](InputIt a, InputIt b){return *a < *b;});
		*d_first = **(window_contents.begin() + (size_t)((high - low)*quantile));
	       ++d_first;
		++t_first;	       
	}
	return d_first;
}

}

#endif
