## Q1 Gradient of Log Likelihood 
#### For simplicity, a single point the logistic regression likelihood is: (Avoids notations with i, j) ####
Derivative of ln of sigmoid function:<br/>


1. $LL(θ) =y*ln\,\sigma(w*x)+(1−y) * ln\,(1−\sigma(w*x))$
<br/><br/>$\dfrac{\delta LL(θ)}{\delta w} = y*\dfrac{\delta}{\delta w}ln\,\sigma(w*x)+(1−y)*\dfrac{\delta}{\delta w}ln\,(1−\sigma(w*x))$     
<br/>Chain Rule:<br/><br/>
  $\dfrac{\delta ln(1- \sigma(w*x)}{\delta w} = \dfrac{\delta f}{\delta u}*\dfrac{\delta u}{\delta w} = -\dfrac{1}{1-u}*\dfrac{\delta\sigma(w*x)}{\delta w}$ , where $u = \sigma(w*x), f = ln(1-u)$ 
<br/><br/>  
$\dfrac{\delta ln(\sigma(w*x)}{\delta w} = \dfrac{\delta f}{\delta u}*\dfrac{\delta u}{\delta w} = \dfrac{1}{u}*\dfrac{\delta\sigma(w*x)}{\delta w}$ , where $u = \sigma(w*x), f = ln(u)$ 
    
<br/> 
Therefore:
    
2. $\dfrac{\delta LL(θ)}{\delta w} = \bigg[\dfrac{y}{\sigma(w*x)}-\dfrac{1-y}{1-\sigma(w*x)}\bigg]*\dfrac{\delta\sigma(w*x)}{\delta w}$
<br/><br/>Chain Rule:<br/><br/>
    $\dfrac{\delta\sigma(w*x)}{\delta w}= \dfrac{\delta}{\delta w(1+e^{-wx})}= \dfrac{\delta g}{\delta p}*\dfrac{\delta p}{\delta w}=\dfrac{1}{p^2}*x*e^{-wx}$ , where $p = 1+e^{-wx}, g = p^-1 $<br/> <br/><br/>

Combining all Variables

3. $\dfrac{\delta LL(θ)}{\delta w} = \bigg[\dfrac{y}{\sigma(w*x)}-\dfrac{1-y}{1-\sigma(w*x)}\bigg]*\dfrac{\delta\sigma(w*x)}{\delta w} = \dfrac{y-\sigma(w*x)}{\dfrac{x*e^{-wx}}{(1+e^{-wx})^2}}*\dfrac{x*e^{-wx}}{(1+e^{-wx})^2}*x$ = $\boldsymbol{\big[ y-\sigma(w*x) \big]*x}$ 


Updating Equation 

For iteration i: 

$w_j^{new}= w_j^{old} - \alpha*\sum_{i=1}^{n} \big[ y^i -\sigma(w*x_j^i) \big]*x_j $

# logistic_regression
