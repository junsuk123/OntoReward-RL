function lp = gaussianLogPdf2D(x,mu,sigma)
e=x-mu;
lp=-0.5*sum(e.^2,2)/(sigma^2)-2*log(sigma)-log(2*pi);
end
