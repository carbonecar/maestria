from math import pi
from typing import TypeVar
from hypothesis import given
from hypothesis import strategies as st

def media(a: float,b: float,c:float) -> float:
    return (a+b+c)/3

#
# Ejercicio 2
#  Definir la funcion sumaMonedas (int, int, int, int, int) -> int tal que sumaMonedas(a, b, c, d, e)
#  es la suma de los valores de las monedas de 1,2,5,10,20 euros.

def sumaMonedas(a_1: int,b_2:int, c_5:int,d_10:int,e_20:int) -> int:
    return a_1*1+b_2*2+c_5*5+d_10*10+e_20*20;

# Ejercicio 3
#Definir una cuncion volumen esfera (float)-float 
def volumen_esfera(r: float) -> float:
    return 4/3*pi*r

def areaDeCroronaCircular(r1: float, r2:float) -> float:
    return pi*r2**2-(pi*r1**2)

def ultimoDigito(numero: int)-> int:
    return numero%10

def maxTres(a: int, b:int, c:int)->int:
    max=a
    if(max<b): 
        max=c
    if(max<c):
        max=c
    return max


