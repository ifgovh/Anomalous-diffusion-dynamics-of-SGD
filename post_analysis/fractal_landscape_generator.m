% Fractal landscape program generator
function [x,y,z] = fractal_landscape_generator(ii)

n = 10;
rugosite = 3;
niv_base = 0;
niv_eau = -0.1;
aff_Etape = false;
bord_aleatoire = false;
bord_lisse = true;
aff_altitude = false;
z = -ones(2^n+1,2^n+1);
r = rugosite;
if bord_aleatoire == false 
    
    z(1,1) = niv_base;
    z(1,2^n+1) = niv_base;
    z(2^n+1,1) = niv_base;
    z(2^n+1,2^n+1) = niv_base; 
    
else    
    
    z(1,1) = (rand-0.5)*r + niv_base;
    z(1,2^n+1) = (rand-0.5)*r + niv_base;
    z(2^n+1,1) = (rand-0.5)*r + niv_base;
    z(2^n+1,2^n+1) = (rand-0.5)*r + niv_base; 
    
end
if (aff_Etape == false && n == 0) | aff_Etape == true
    
    [X Y] = meshgrid(1:2,1:2);
    surf(X,Y,[ z(1,1), z(1,2^n+1); z(2^n+1,1), z(2^n+1,2^n+1)],'FaceColor','interp','FaceLighting','phong');    
    axis tight, axis off;
    shading interp;
    colormap('winter');
    camlight('headlight');
    set(gcf,'Color',[0 0 0]), set(gca,'Color',[0 0 0]);
    pause(1);
    
end
if n > 0
    
    for nc = 1:n
        
        nb_pt = 2^nc;
        pas = 2^n/nb_pt;
        nb_new_pt = nb_pt/2;
        Ml = ones(nb_new_pt,nb_new_pt);
        Mc = ones(nb_new_pt,nb_new_pt);
        for j = 1:nb_new_pt
            
            for i = 1:nb_new_pt
                
                if i ~= 1 | j ~= 1
                    
                    l = (2*pas)*(j-1) + pas + 1;
                    c = (2*pas)*(i-1) + pas + 1;
                    
                elseif i == 1 & j == 1
                    
                    l = pas + 1;
                    c = l;
                    
                end
                pt_ne = [l+1,c-1];
                pt_no = [l-1,c-1];
                pt_se = [l+1,c+1];
                pt_so = [l-1,c+1];
                while (pt_ne(1,1)+1) <= 2^n+1 & (pt_ne(1,2)-1) >= 1 & z(pt_ne(1,1),pt_ne(1,2)) == -1
                    
                    pt_ne(1,1) = pt_ne(1,1)+1;
                    pt_ne(1,2) = pt_ne(1,2)-1;
                    
                end
                while (pt_no(1,1)-1) >= 1 & (pt_no(1,2)-1) >= 1 & z(pt_no(1,1),pt_no(1,2)) == -1
                    
                    pt_no(1,1) = pt_no(1,1)-1;
                    pt_no(1,2) = pt_no(1,2)-1;
                    
                end
                while((pt_se(1,1)+1) <= 2^n+1 & (pt_se(1,2)+1) <= 2^n+1 & z(pt_se(1,1),pt_se(1,2)) == -1)
                    
                    pt_se(1,1) = pt_se(1,1)+1;
                    pt_se(1,2) = pt_se(1,2)+1;
                    
                end
                while((pt_so(1,1)-1) >= 1 & (pt_so(1,2)+1) <= 2^n+1 & z(pt_so(1,1),pt_so(1,2)) == -1)
                    
                    pt_so(1,1) = pt_so(1,1) - 1;
                    pt_so(1,2) = pt_so(1,2) + 1;
                    
                end
                moy = (z(pt_ne(1,1),pt_ne(1,2)) + z(pt_no(1,1),pt_no(1,2)) + z(pt_se(1,1),pt_se(1,2)) + z(pt_so(1,1),pt_so(1,2)))/4;
                z(l,c) =  moy + (rand-0.5)*r;
                Ml(i,j) = l;
                Mc(i,j) = c;
            end
        end
        for i = 1:nb_new_pt
            
            for j = 1:nb_new_pt
                l = Ml(i,j);
                c = Mc(i,j);
                if z(l-pas,c) == -1
                    
                    if (l-pas) == 1
                        
                        moy = (z(l,c)+z(l-pas,c+pas)+z(l-pas,c-pas))/3;
                        
                    else
                        
                        moy = (z(l,c)+z(l-2*pas,c)+z(l-pas,c+pas)+z(l-pas,c-pas))/4;
                        
                    end
                    
                    z(l-pas,c) = moy + (rand-0.5)*r;
                    
                end
                if z(l,c+pas) == -1
                    
                    if (c+pas) == 2^n+1
                        
                        moy = (z(l,c)+z(l-pas,c+pas)+z(l+pas,c+pas))/3;
                        
                    else
                        
                        moy = (z(l,c)+z(l-pas,c+pas)+z(l+pas,c+pas)+z(l,c+2*pas))/4;
                        
                    end
                    
                    z(l,c+pas) = moy + (rand-0.5)*r;
                    
                end
                if z(l+pas,c) == -1
                    
                    if (l+pas) == 2^n+1
                        
                        moy = (z(l,c)+z(l+pas,c-pas)+z(l+pas,c+pas))/3;
                        
                    else
                        
                        moy = (z(l,c)+z(l+pas,c-pas)+z(l+pas,c+pas)+z(l+2*pas,c))/4;
                        
                    end
                    
                    z(l+pas,c) = moy + (rand-0.5)*r;
                    
                end
                if z(l,c-pas) == -1
                    
                    if (c-pas) == 1
                        
                        moy = (z(l,c)+z(l-pas,c-pas)+z(l+pas,c-pas))/3;
                        
                    else
                        
                        moy = (z(l,c)+z(l-pas,c-pas)+z(l+pas,c-pas)+z(l,c-2*pas))/4;
                        
                    end
                    
                    z(l,c-pas) = moy  + (rand-0.5)*r;
                    
                end
            end
            
        end
        
        x = ones(1,nb_pt+1);
        y = ones(1,nb_pt+1);
        for k = 0:nb_pt
            
            x(1,k+1) = k*(1/nb_pt)+1;
            
        end
        for k = 0:nb_pt
            
            y(1,k+1) = k*(1/nb_pt)+1;
            
        end
        [X,Y] = meshgrid(x,y);
        Z = ones(nb_pt+1,nb_pt+1);
        m = [0,1];
        for j = 1:2^n+1
            
            for i = 1:2^n+1
                
                if z(j,i) ~= 1
                    m(1,1) = m(1,1)+1;
                    Z(m(1,2),m(1,1)) = z(j,i);
                    
                end
                
            end
            
            if m(1,1) == nb_pt+1
                
                m(1,2) = m(1,2)+1;
                
            end
            
            m(1,1) = 0;
            
        end
        if bord_lisse == true & bord_aleatoire == false
            
            for i = 1:nb_pt+1
                
                if Z(i,1) > niv_base
                    
                    Z(i,1) = niv_base;
                    
                end
                
                if Z(1,i) > niv_base
                    
                    Z(1,i) = niv_base;
                    
                end
                
                if Z(nb_pt+1,i) > niv_base
                    
                    Z(nb_pt+1,i) = niv_base;
                    
                end
                
                if Z(i,nb_pt+1) > niv_base
                    
                    Z(i,nb_pt+1) = niv_base;
                    
                end
                
            end
            
        end
        if niv_eau ~= false
            
            for i = 1:nb_pt+1
                
                for j = 1:nb_pt+1
                    
                    if Z(j,i) < niv_eau
                        
                        Z(j,i) = niv_eau;
                        
                    end
                end
                
            end
            
        end
        if (aff_Etape == false && n == nc) | (aff_Etape == true)
            figure('visible','off')
            imagesc(Z)
            %surf(X,Y,Z,'FaceColor','interp','FaceLighting','phong');            
            axis tight, axis off;
            %shading interp;
            colormap('jet');
            %camlight('headlight');
            set(gcf,'Color',[0 0 0]), set(gca,'Color',[0 0 0]);
            if aff_altitude == true
                colorbar
            end
            %pause(1);
            saveas(gcf,['test',num2str(ii),'.jpg'])
        end
        r = r/2;
    end
    
end