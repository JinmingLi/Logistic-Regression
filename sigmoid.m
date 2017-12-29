function g = sigmoid(z)

g = zeros(size(z));

g = 1./(1 + exp(-z));
%sigmf(z,[1,0]);

end
