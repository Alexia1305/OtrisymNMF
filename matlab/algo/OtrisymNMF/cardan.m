function roots = cardan(a,b,c,d)
    if a == 0
        if b == 0
            root1 = -d / c;
            roots = root1;
            return;
        end
        delta = c^2 - 4 * b * d;
        root1 = (-c + sqrt(delta)) / (2 * b);
        root2 = (-c - sqrt(delta)) / (2 * b);
        if root1 == root2
            roots = root1;
        else
            roots = [root1, root2];
        end
        return;
    end

    p = -(b^2 / (3 * a^2)) + c / a;
    q = ((2 * b^3) / (27 * a^3)) - ((9 * c * b) / (27 * a^2)) + (d / a);
    delta = -(4 * p^3 + 27 * q^2);

    if delta < 0
        u = (-q + sqrt(-delta / 27)) / 2;
        v = (-q - sqrt(-delta / 27)) / 2;
        if u < 0
            u = -(-u)^(1 / 3);
        elseif u > 0
            u = u^(1 / 3);
        else
            u = 0;
        end
        if v < 0
            v = -(-v)^(1 / 3);
        elseif v > 0
            v = v^(1 / 3);
        else
            v = 0;
        end
        root1 = u + v - (b / (3 * a));
        roots = root1;
        return;
    elseif delta == 0
        if p == 0 && q == 0
            root1 = 0;
            roots = root1;
        else
            root1 = (3 * q) / p;
            root2 = (-3 * q) / (2 * p);
            roots = [root1, root2];
        end
        return;
    else
        epsilon = -1e-300;
        phi = acos(-q / (2 * sqrt(-27 / (p^3 + epsilon))));
        z1 = 2 * sqrt(-p / 3) * cos(phi / 3);
        z2 = 2 * sqrt(-p / 3) * cos((phi + 2 * pi) / 3);
        z3 = 2 * sqrt(-p / 3) * cos((phi + 4 * pi) / 3);
        root1 = z1 - (b / (3 * a));
        root2 = z2 - (b / (3 * a));
        root3 = z3 - (b / (3 * a));
        roots = [root1, root2, root3];
        return;
    end
end
