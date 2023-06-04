# IAPC 

## <p align="center">Towards Source-free Domain Adaptive Semantic Segmentation via Importance-aware and Prototype-contrast Learning
  
<div align="center">
 
[Paper]()
  
<img src="fig/architecture.png" />
 
 </div>
 
 Code for our paper "Towards Source-free Domain Adaptive Semantic Segmentation via Importance-aware and Prototype-contrast Learning"
 
In this paper, we propose an end-to-end source-free domain adaptation semantic segmentation method via Importance-Aware and Prototype-Contrast (IAPC) learning. The proposed IAPC framework effectively extracts domain-invariant knowledge from the well-trained source model and learns domain-specific knowledge from the unlabeled target domain.



## Code

### TODO List
- [ ] Code for "GTA5-to-Cityscapes"
- [ ] Code for "Synthia-to-Cityscapes"
- [ ] Checkpoints and prediction maps

## Results
<div align="left">
<table>
  <tr>
      <td></td> 
      <th>Model</th> 
      <th>mIoU</th> 
      <th>mIoU*</th>
  </tr>
  <tr>
      <td rowspan="2">GTA5-to-Cityscapes</td>    
      <td ><a href="#" target="_blank">Source Only</a></td>  
      <td >36.6</td> 
      <td >-</td>  
   </tr>
   <tr>
      <td ><a href="#" target="_blank">IAPC</a></td> 
      <td >49.4</td> 
      <td >-</td>
   </tr>
   <tr>
      <td rowspan="2">Synthia-to-Cityscapes</td>    
      <td ><a href="#" target="_blank">Source Only</a></td>  
      <td >35.2</td> 
      <td >40.5</td>  
   </tr>
   <tr>
      <td ><a href="#" target="_blank">IAPC</a></td> 
      <td >45.3</td> 
      <td >52.6</td>
   </tr>
</table>
   </div>
