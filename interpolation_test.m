% ----------> X
% |
% | interpolation grid
% |
% Y
sample_size = 500;  sa = 1; sb = 500;
interp_size = 5000; ia = 1; ib = 500;
sampleX = linspace(sa, sb, sample_size);
sampleY = linspace(sa, sb, sample_size);
interpX = linspace(ia, ib, interp_size);
interpY = linspace(ia, ib, interp_size);

[gridSX, gridSY] = meshgrid(sampleX, sampleY);
[gridIX, gridIY] = meshgrid(interpX, interpY);

sampleValue = zeros(sample_size, sample_size);
for i=1:sample_size
    for j=1:sample_size
        sampleValue(i,j) = i + j;
    end
end

interpValue = interp2(gridSX, gridSY, sampleValue, gridIX, gridIY, 'linear', 0);


