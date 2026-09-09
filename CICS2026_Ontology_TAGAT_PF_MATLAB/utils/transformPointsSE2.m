function out = transformPointsSE2(pts,T)
c = cos(T(3)); s = sin(T(3));
R = [c -s; s c];
out = pts*R.' + T(1:2);
end
