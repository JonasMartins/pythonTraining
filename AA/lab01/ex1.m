### Importar dados

data = load('ex1data1.txt');

x = data(:,1);
y = data(:,2);

### Regressao univariada
### Gradiente descendente
### x*w = y
### w???
### w é o vetor de pesos que queremos achar

#Adicionando intercepto
x = [ones(size(x,1),1) x];

#inicializando os pesos
w = zeros(2,1);

#Passo de aprendizagem
alpha = 0.001;

#Quantidade de iteraçoes do aprendizado
nEpocas = 1000;

n = size(x,1)

eqm = []

for i=1:nEpocas
  ind = randperm(n);
  x = x(ind,:);
  y = y(ind);
  erro_total = 0
  for j=1:n
    y_barra = x(j,:)*w;
    erro = y(j) - y_barra;
    erro_total = erro_total + erro*erro;
    w(1) = w(1) + alpha*erro;
    w(2) = w(2) + alpha*erro*x(j,2);
  endfor
  eqm = [eqm erro_total/n] 
endfor

plot(1:nEpocas, eqm)

#plot(x(:,2), y,'bo')
#hold on
#plot(x(:,2), x*w,'rx')

#plot(x,y,'*');




