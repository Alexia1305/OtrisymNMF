function xopt = cardan_depressed(c3,c2,c1,c0)
    if abs(c3)<1e-6
        xopt = -c0/c1;
    else

        %Si c2=0
        np=c1/c3;
        nq=c0/c3;

        Delta = 4*np^3+27*nq^2;
        d     = 0.5*(-nq+sqrt(Delta/27));

        % For values where Delta is <= 0
        if Delta <=0 %-> 3 solutions réelles distinctes ou une solution multiple mais toutes réelles
            r3        = 2*(abs(d)^(1/3)); %racine cubique du module de d multipliée par 2
            th3       = (atan2(imag(d),real(d)))/3; %angle(dneg)/3;       %argument de d divisé par 3
            x         = (r3)*[cos(th3) cos(th3+(2*pi/3))];

            [~,ind]   = min(x.^4/4+(np)*x.^2/2 + (nq)*x,[],2);
            xopt      = x(ind);
        else
            d2pos     = 0.5*(-nq-sqrt(Delta/27));
            xopt      = sign(d)*(abs(d))^(1/3) + sign(d2pos)*(abs(d2pos))^(1/3);
        end
    end
end