X = rand(1,300)*20;
Y = X * 2 + randn(1,300)
Y2 = X * 2 +randn(1,300) - 4

for a = 1:300
   if X(a)<=4 && X(a)>= 2
       Y(a) = Y(a) + 4;
       Y2(a) = Y2(a) + 4;
   end
   
   if X(a)<=16 && X(a)>= 14
       Y(a) = Y(a) - 8;
       Y2(a) = Y2(a) - 8;
   end
   if X(a)<=10 && X(a)>=9
       Y(a) = Y(a) -2;
       Y2(a) = Y2(a) -2;
   end
end

figure
hold on
scatter(X,Y, 'r')
scatter(X,Y2, 'b')

for i = 1:300
   fprintf('%f, %f, 0\n',X(i),Y(i)); 
   fprintf('%f, %f, 1\n',X(i),Y2(i)); 
end