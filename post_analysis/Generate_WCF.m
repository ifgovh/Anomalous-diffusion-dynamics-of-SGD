function [WCF] = Generate_WCF(landa, H, N)
%{
Function for generating a parametric fractal test signal called Weierstrass
Cosine Function (WCF). This function may be very useful to assess fractal
dimension measures.
For more information see: Margaos and Sun, 'Measuring the Fractal Dimension
of Signals: Morphological Covers and Iterative Optimization'.
INPUT: 
    landa: a parameter in the equation. landa > 1. If landa is an integer,
    then, the WCF is periodic with period one. 
    H: a parameter of the equation. 0 < H < 1. D = 2 - H, being D de
    fractal dimension of WCF signal. 
    N: the number of samples employed in the generation of WCF.
OUTPUT:
    WCF: the parametric fractal test signal WCF.
PROJECT: Research Master in signal theory and bioengineering - University of Valladolid
DATE: 19/12/2013
AUTHOR: Jesús Monge Álvarez
%}
%% Checking the ipunt parameters:
control = ~isempty(landa);
assert(control,'The user must introduce the landa parameter (first inpunt).');
control = ~isempty(H);
assert(control,'The user must introduce the H parameter (second inpunt).');
control = ~isempty(N);
assert(control,'The user must introduce the number of samples they want (third inpunt).');
%% Processing:
% Generation:
t = linspace(0,1,N); %N samples. 
% Equation:
% WCF(t) = sum from k = 0 up to Kmax of: landa^-kH * cos(2*pi*(landa^k)*t).
% Kmas is usually chosen such 2*pi*(landa^k) <= 10^12. 
% More or less, the above is fulfilled for: Kmax = 9. 
Kmax = 10;
aux = NaN(1,Kmax);
aux2 = NaN(Kmax,N);
for k = 1:Kmax
    aux(k) = landa^((-1)*(k*H));
    aux2(k,:) = aux(k).*cos(2*pi*(landa^k)*t);
end
WCF = sum(aux2);
