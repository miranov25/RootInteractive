# RDataFrame interface - domain specific language

ROOT RDataFrame provides a modern, sophisticated interface for analysing data stored in TTree , CSV and other data 
formats in C++ or Python.
In our RDataFrame_array.py we provide an interface for creating C++ template functions, C++ 
libraries and RDataFrame columns using a domain-specific language similar to the old type tree->Draw query language.

We extend C++ with Python syntax for array slicing and projection. Internally, Python ast, cppyy.ll and 
RDataFrame type information are used to parse Python like a functional syntax.

https://github.com/miranov25/RootInteractive/blob/647b43e9c80f1c469f9c64c2cfff11744d0028df/RootInteractive/Tools/RDataFrame/RDataFrame_Array.py#L536-L544


## Example single Define
* 
```
    rdf=makeDefine(f"qPt","track[:].getQ2Pt()",rdf,rdfLib,3,0x2)
    pprint.pprint(rdfLib["qPt"])
```
```
 'qPt': {'code': 'track[:].getQ2Pt()',
         'dependencies': ['track'],
         'implementation': 'ROOT::VecOps::RVec<float> '
                           'qPt(ROOT::VecOps::RVec<o2::tpc::TrackTPC> '
                           '&track){'
                           '    ROOT::VecOps::RVec<float> result(track.size() '
                           '- 0);'
                           '    for(size_t i=0; i<track.size() - 0; i++){\n'
                           '        result[i] = track[0+i*1].getQ2Pt();\n'
                           '    }'
                           '    '
                           '    return result;'
                           '} ',
         'name': 'qPt',
         'type': 'ROOT::VecOps::RVec<float>'},
```

## Projection example queries
```
array2D                   -> not suported yet - range has to be specified
array2D[:,:]              -> 2D array
array2D[index0]           -> 1D array
array2D[:,index1]         -> 1D array
array2D[index0,index1]    -> scalar
```
```
array2D[:,index1]-array1D[:]
```



## Join like iterators
* [left, right, upper bound, nearest, in range,stat in range (rolling)]
* in pandas https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.merge_asof.html

```
join=("left","right")
index=([indices],0)
```
left="track", right="

```
makeDefineJoin(<name>,<function>,<dataFrame>,[left],[rights])
```

`ROOT::VecOps::RVec<float>
getVertexZ(ROOT::VecOps::RVec<int> &trackIndices,
           ROOT::VecOps::RVec<o2::dataformats::Vertex<float>> &collisions) {
  ROOT::VecOps::RVec<float> VertexZ(trackIndices.size());
  int index;
  for (std::size_t i = 0; i < trackIndices.size(); i++) {
    index = trackIndices[i];
    if (index >= 0 && index < static_cast<int>(collisions.size())) {
      VertexZ[i] = collisions[index].getZ();
    } else {
      VertexZ[i] = -1;
    }
  }
  return VertexZ;
}`

## Join like iterator
* [left, right, upper bound, nearest, in range,stat in range]
* in pandas https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.merge_asof.html
* 