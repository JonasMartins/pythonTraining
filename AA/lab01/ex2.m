### Importar dados

data = load('ex1data2.txt');

x = data(:,1:2);
y = data(:,3);

[n m] = size(x);

### Regressao univariada
### Gradiente descendente
### x*w = y
### w???
### w é o vetor de pesos que queremos achar

#Adicionando intercepto
x = [ones(n,1) x];

#inicializando os pesos
w = zeros(m+1,1);

#Passo de aprendizagem
alpha = 0.01;

#Quantidade de iteraçoes do aprendizado
nEpocas = 100;

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
    w = w + (alpha*erro*x(j,:))';
  endfor
  eqm = [eqm erro_total/n] 
endfor

plot(1:nEpocas, eqm)

#plot(x(:,2), y,'bo')
#hold on
#plot(x(:,2), x*w,'rx')

#plot(x,y,'*');

I = eye(m,m)
I(1,1) = 0
w=pinv(x'*x + lamda*I)*x'*y


