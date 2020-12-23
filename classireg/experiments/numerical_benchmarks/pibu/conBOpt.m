classdef conBOpt < handle
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Implementation of constrained Bayesian optimization
    %
    % Uses GP regression for the cost function and binary GP classification
    % for the success function. The implemented acquisition function combines
    % the probability of improvement with boundary uncertainty.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    properties
        t;         % grid that defines search space
        n;         % input dimension
        e;         % PI trade-off parameter (default 0)
        bOffset;   % offset of points considered on boundary (default 0.1)
        bThreshold;% threshold of ignored boundary variance (default 0)
        X,Y,YS;    % dataset consisting of input, reward, success
        gpr,gpc;   % hyper parameter for gp regression/classification
        trainHypR; % train hyper parameter of regression gp
        trainHypC; % train hyper parameter of classification gp
        optM;      % method used for optimizing the acquisition function
                   % 1: grid search, 2: multistart optimization
        verbose;   % controls the output
        
        t1,t2,t3,yc,ysc2,fcmu,fcs2;
        yr,sr2; % predicted mean and variance
        PI,BU,a,fx,x,y_gt; % variables for plots
        statBCx, statBCy, statNOF, statBCidx; % variables for stats
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods
        function c = conBOpt(n,t,e,bOffset,bThreshold,verbose,optM,ellC,sf2C,ellR,sf2R,snR)
            c.n = n;
            c.t = t;
            c.bOffset = bOffset;
            c.bThreshold = bThreshold;
            c.e = e;
            c.verbose = verbose;
            c.optM = optM;
            c.X = []; c.Y = []; c.YS = [];
            
            % init classification gp
            c.trainHypC = false;
            c.gpc.meanfunc = @meanConst;  c.gpc.hyp.mean = -7; % amarco: why -7?
            c.gpc.covfunc = {@covSEiso};  c.gpc.hyp.cov = log([ellC, sf2C]);
            c.gpc.likfunc = @likLogistic; c.gpc.inf = @infLaplace;
            
            % init regression gp
            c.trainHypR = false;
            c.gpr.meanfunc = [];
            c.gpr.covfunc = @covSEiso;  c.gpr.hyp.cov = log([ellR; sf2R]);
            c.gpr.likfunc = @likGauss;  c.gpr.hyp.lik = log(snR);
            
            % precompute matrices for plotting
            if (n==2)
                c.t1 = reshape(c.t(:,1),sqrt(size(c.t,1)),sqrt(size(c.t,1)));
                c.t2 = reshape(c.t(:,2),sqrt(size(c.t,1)),sqrt(size(c.t,1)));
            end
            if (n==3)
                m = round(power(size(c.t,1),1/3));
                c.t1 = reshape(c.t(:,1),m,m,m);
                c.t2 = reshape(c.t(:,2),m,m,m);
                c.t3 = reshape(c.t(:,3),m,m,m);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [x,isConv] = selectNextPoint(c)

            % fprintf('Here\n')
            % predict with classifier
            if c.trainHypC && (sum(c.YS==-1)>3)
                c.gpc.hyp = minimize(c.gpc.hyp, @gp, -200, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS);
            end
            [c.yc, c.ysc2, c.fcmu, c.fcs2] = gp(c.gpc.hyp, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS, c.t);
            
            % fprintf('Here2\n')
            % predict with regression model
            if c.trainHypR && (size(c.X,1)>10)
                c.gpr.hyp = minimize(c.gpr.hyp, @gp, -200, @infExact, c.gpr.meanfunc, c.gpr.covfunc, c.gpr.likfunc, c.X(c.YS==1,:), c.Y(c.YS==1));
            end
            [c.yr, c.sr2] = gp(c.gpr.hyp, @infExact, c.gpr.meanfunc, c.gpr.covfunc, c.gpr.likfunc, c.X(c.YS==1,:), c.Y(c.YS==1), c.t);
            
            % fprintf('Here3\n')

            %% compute acqusition function on grid t
            idxIR = find(c.yc>c.bOffset);    % points in inner region
            [~, idxB] = sort(abs(c.fcmu));
            idxB = find(c.yc>-c.bOffset & c.yc<c.bOffset); % points on boundary
            [~, b]=max(c.fcs2(idxB));
            
            [yMax, iMax] = max(c.Y(c.YS==1));
            c.PI = zeros(size(c.t,1),1);
            c.PI(idxIR) = normcdf((c.yr(idxIR)-yMax-c.e)./sqrt(c.sr2(idxIR))); % prob. of improvement
            
            % keyboard;

            c.BU = zeros(size(c.t,1),1);
            c.BU(idxB) = sqrt(c.fcs2(idxB))/exp(c.gpc.hyp.cov(end));
            
            c.BU(c.BU<c.bThreshold)=0; % ignore boundary uncertainty below a certain threshold
            
            c.a = zeros(size(c.t,1),1); % acquisition function
            c.a = c.BU + c.PI;
            
            if (c.optM==1)
                %% grid search
                [c.fx,k] = max(c.a);
                c.x = c.t(k,:);
                x = c.x;
            elseif (c.optM==2)
                %% multistart optimization
                [c.x,c.fx] = c.optimizeAcquistion();
                x = c.x;
            end
            
            %% check for convergence
            isConv=false;
            if length(c.X)>0 & (min(sum(abs(c.X-repmat(x,size(c.X,1),1)),2))<1e-3)
                isConv=true;
                display(['conBOpt converged after ', num2str(length(c.X)),' iterations']);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function addDataPoint(c,x,y,ys)
            % make sure x is a row vector
            if size(x,1)>size(x,2)
                x = x';
            end
            % check if datapoint is already in database (causes problem with GP)
            if length(c.X)>0 & (min(sum(abs(c.X-repmat(x,size(c.X,1),1)),2))==0)
                warning('datapoint is already in database');
            end
            
            % add datapoint
            c.X = [c.X; x];
            c.Y = [c.Y; y];
            c.YS = [c.YS; ys];
            
            % bookkepping
            Ytmp = c.Y;
            Ytmp(c.YS==-1) = -inf;
            [~,bcIdx] = max(Ytmp);
            c.statBCx = [c.statBCx; c.X(bcIdx,:)];
            c.statBCy = [c.statBCy; c.Y(bcIdx,:)];
            c.statNOF = [c.statNOF; sum(c.YS==-1)];
            c.statBCidx = [c.statBCidx; bcIdx];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [x, fx] = optimizeAcquistion(c)
            lb = min(c.t);
            ub = max(c.t);
            [yMax] = max(c.Y(c.YS==1));
            sampleX0 = @() lb+rand(1,c.n).*(ub-lb);
            
            if (c.verbose<3)
                options = optimoptions('fmincon','GradObj','on','Display','off');
            else
                options = optimoptions('fmincon','GradObj','on','Display','iter-detailed');
            end
            
            %% optimize prob. of improvement inside the success region
            f = @(x) c.nPIfun(yMax,x);
            con = @(x) c.PIcon(x);
            [~, l] = max(c.a);
            x = c.t(l,:);
            fx = -c.a(l);
            for i = 1:1
                [~,k] = max(c.PI);
                x0 = c.t(k,:);
                count_ = 0;
                while con(x0)>0 && count_ < 1e3
                    % fprintf("While loop -> Sampling 1...\n")
                    % keyboard;
                    x0 = sampleX0();     % sample feasible start point
                    count_ = count_ + 1;
                end
                if count_ == 1e3
                    fprintf('While loop got stuck\n');
                    keyboard;
                end
                [x_i,f_i,optflag]=fmincon(f,x0,[],[],[],[],lb,ub,con,options);
                if (f_i<fx & optflag>0)
                    x = x_i;  fx = f_i; % pick best candidate
                end
            end
            
            %% optimize BU on the boundary of the success region
            f = @(x) c.nBUfun(x);
            con = @(x) c.BUcon(x);
            if (max(c.BU)>0)
                for i = 1:1
                    [~,k] = max(c.BU);
                    x0 = c.t(k,:);
                    count_ = 0;
                    while sum(con(x0)>0)>0 && count_ < 1e3
                        % fprintf("Sampling 1...")
                        x0 = sampleX0();     % sample feasible start point
                        count_ = count_ + 1;
                    end
                    if count_ == 1e3
                        fprintf('While loop got stuck\n');
                        keyboard;
                    end
                    [x_i,f_i,optflag]=fmincon(f,x0,[],[],[],[],lb,ub,con,options);
                    if (optflag<0)
                        optflag
                        % keyboard;
                    end
                    if (f_i<fx & optflag>0)
                        x = x_i;  fx = f_i; % pick best candidate
                    end
                end
            end
            
            fx = -fx;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % This function computes the negative probability of improvement and
        % the corresponding gradient with respect to the input
        function [f,dfdx] = nPIfun(c,yMax,x)
            X = c.X(c.YS==1,:);
            Y = c.Y(c.YS==1);
            [yr, sr2] = gp(c.gpr.hyp, @infExact, c.gpr.meanfunc, c.gpr.covfunc, c.gpr.likfunc,X,Y,x);
            K = covSEiso(c.gpr.hyp.cov,X,X)+eye(size(X,1))*exp(2*c.gpr.hyp.lik);
            alpha = inv(K)*Y;
            k = covSEiso(c.gpr.hyp.cov,x,X);
            ell2 = exp(c.gpr.hyp.cov(1)).^2;
            L = eye(size(X,2))./ell2;
            Xt = repmat(x,size(X,1),1)-X;
            dyrdx = -L*Xt'*(k'.*alpha);
            dkdx = zeros(size(X,1),c.n);
            for j=1:size(X,1)
                dkdx(j,:) = -L*Xt(j,:)'*k(j);
            end
            dsr2dx = -2*k*inv(K)*dkdx;
            z = (yr-yMax-c.e)./sqrt(sr2);
            f = -normcdf(z);
            dzdx = dyrdx/sqrt(sr2) -(yr-yMax-c.e)*0.5*dsr2dx'*sr2^(-3/2);
            dfdx = -normpdf(z)*dzdx;
            f = f;
            dfdx = dfdx;
        end
        
        % This function describesan inequality constraint that measures if an
        % input is inside the success region
        function [h,heq] = PIcon(c,x)
            yc = gp(c.gpc.hyp, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS, x);
            h = -yc + c.bOffset;
            heq = [];
        end
        
        % This function computes the negative boundary uncertainty and
        % the corresponding gradient with respect to the input
        function [f,dfdx] = nBUfun(c,x)
            [~, ~, ~, fcs2] = gp(c.gpc.hyp, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS, x);
            K = covSEiso(c.gpc.hyp.cov,c.X,c.X);
            k = covSEiso(c.gpc.hyp.cov,x,c.X);
            ell2 = exp(c.gpc.hyp.cov(1)).^2;
            L = eye(size(c.X,2))./ell2;
            Xt = repmat(x,size(c.X,1),1)-c.X;
            [~, ~, fcmu] = gp(c.gpc.hyp, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS, c.X); % amarco: fcmu is the predictive mean
            p = sigmoid(fcmu);
            W = diag(p.*(1-p));

            try
                sr22 = covSEiso(c.gpc.hyp.cov,x,x) - k*inv(K + inv(W))*k';
            catch
                keyboard;
            end
            dkdx = zeros(size(c.X,1),c.n);
            for j=1:size(c.X,1)
                dkdx(j,:) = -L*Xt(j,:)'*k(j);
            end
            dsr2dx = -2*k*inv(K+inv(W))*dkdx;
            f = -sqrt(fcs2)/exp(c.gpc.hyp.cov(end));
            dfdx = -dsr2dx'/(2*sqrt(fcs2)*exp(c.gpc.hyp.cov(end)));
        end
        
        % This functions describes two inequality constraints that measure if an
        % input is on the boundary of the classification gp
        function [h,heq] = BUcon(c,x)
            yc = gp(c.gpc.hyp, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS, x);
            h = yc-c.bOffset;
            h = [h;-yc-c.bOffset];
            heq = [];
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [xBc,yBc,out] = stats(c)
            [c.yc, c.ysc2, c.fcmu, c.fcs2] = gp(c.gpc.hyp, c.gpc.inf, c.gpc.meanfunc, c.gpc.covfunc, c.gpc.likfunc, c.X, c.YS, c.t);
            
            if (c.verbose>0)
                fprintf( '\n### Statistics ###\n');
                fprintf( '- amount of samples         : %d\n',length(c.Y));
                fprintf( '- amount of failures        : %d\n',c.statNOF(end));
                fprintf(['- last candidate            : ',repmat('%6.4f  ',1,c.n),'\n'],c.X(end,:));
                fprintf( '- last candidate reward     : %f\n',c.Y(end));
                fprintf( '- last candidate success    : %d\n',c.YS(end));
                fprintf(['- best candidate            : ',repmat('%6.4f  ',1,c.n),'\n'],c.statBCx(end,:));
                fprintf( '- best candidate reward     : %f\n',c.statBCy(end) );
                fprintf( '- max boundary uncertainty  : %f\n',max(c.BU));
                fprintf( '- max prob. of improvement  : %f\n',max(c.PI));
            end
            xBc = c.statBCx(end,:);
            yBc = c.statBCy(end);

            out = struct();
            out.statBCx = c.statBCx;
            out.statBCy = c.statBCy;
            out.X = c.X;
            out.Y = c.Y;
            out.statNOF = c.statNOF;

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function plot(c)
            if (c.verbose>0)
                if (c.n==1)
                    %% 1d plotting
                    f = figure(1);clf;hold on;
                    title('Function value');
                    g = shadedErrorBar(c.t,c.yr,2*sqrt(c.sr2));
                    mint = min(c.t(abs(c.yc)<1e-1));
                    maxt = max(c.t(abs(c.yc)<1e-1));
                    edgeL = 0.03;
                    
                    plot(c.X(c.YS==1),c.Y(c.YS==1),'ko');
                    plot(c.X(c.YS==-1),c.Y(c.YS==-1),'rx');
                    plot(c.t,c.yr,'b-');
                    plot(c.X(end),c.Y(end),'.g','MarkerSize',20)
                    axis tight;
                    xlabel('\theta');
                    ylabel('R(\theta)');
                    minLmaxL = get(get(f,'CurrentAxes'),'YLim');
                    minL = .9*minLmaxL(1);
                    maxL = .9*minLmaxL(2);
                    bounds = plot([mint,mint],[minL,maxL],'r');
                    plot([maxt,maxt],[minL,maxL],'r')
                    plot([maxt,maxt-edgeL],[maxL,maxL],'r')
                    plot([mint,mint+edgeL],[maxL,maxL],'r')
                    plot([maxt,maxt-edgeL],[minL,minL],'r')
                    plot([mint,mint+edgeL],[minL,minL],'r')
                    
                    if (length(c.y_gt)>0)
                        gt = plot(c.t,c.y_gt,':','LineWidth',3);
                        legend([g.mainLine, g.patch, bounds, gt],'GP mean','GP 2\sigma','Safety boundary','R(\theta)','Location','SouthEast')
                    else
                        legend([g.mainLine, g.patch],'GP mean','GP 2\sigma','Location','SouthEast')
                    end
                    
                    figure(2);clf;hold on;
                    title('Safety region');
                    shadedErrorBar(c.t,c.fcmu,2*sqrt(c.fcs2));
                    if (exist('c.yS_gt'))
                        plot(c.t,c.yS_gt,'-');
                    end
                    plot(c.X(c.YS==1),c.YS(c.YS==1),'ko');
                    plot(c.X(c.YS==-1),c.YS(c.YS==-1),'rx');
                    plot(c.X(end),c.YS(end),'.g','MarkerSize',20)
                    plot(c.t(abs(c.yc)<1e-1),c.yc(abs(c.yc)<1e-1),'r.','MarkerSize',15);
                    xlabel('\theta');
                    ylabel('S(\theta)');
                    legend('PIBU','Location','SouthEast');
                    axis tight;
                    
                    figure(3);clf;hold on;
                    title('Acquisition function');
                    plot(c.t,c.PI,'-');
                    plot(c.t,c.BU,'r-');
                    plot(c.x,c.fx,'g.','MarkerSize',20);
                    
                    xlabel('\theta');
                    ylabel('a(\theta)');
                    legend('PI','BU','next point');
                    drawnow;
                    if (c.verbose>1)
                        keyboard;
                    end
                elseif (c.n==2)
                    %% 2d plotting
                    figure(1);clf;hold on;
                    title('Function value');
                    tmp = reshape(c.yr,size(c.t1));
                    ps = round(size(c.t1,1)/100);
                    s=surface(c.t1(1:ps:end,1:ps:end),c.t2(1:ps:end,1:ps:end),tmp(1:ps:end,1:ps:end));
                    set(s,'FaceColor','flat','LineStyle','none');
                    plot3(c.X(c.YS==1,1),c.X(c.YS==1,2),c.Y(c.YS==1),'k.','MarkerSize',20);
                    plot3(c.X(c.YS==-1,1),c.X(c.YS==-1,2),c.Y(c.YS==-1)*0+max(c.Y(c.YS==1)),'r.','MarkerSize',20);
                    plot3(c.X(end,1),c.X(end,2),c.Y(end),'g.','MarkerSize',30);
                    if (exist('c.Y_gt'))
                        tmp = reshape(y_gt,size(t1));
                        s=surface(t1(1:ps*5:end,1:ps*5:end),t2(1:ps*5:end,1:ps*5:end),tmp(1:ps*5:end,1:ps*5:end));
                        set(s,'FaceColor','none');
                    end
                    view([-60,55]);
                    grid on;
                    axis tight;
                    drawnow;
                    
                    figure(2);clf;hold on;
                    title('Safety region');
                    tmp = reshape(c.yc,size(c.t1));
                    s=surface(c.t1(1:ps:end,1:ps:end),c.t2(1:ps:end,1:ps:end),tmp(1:ps:end,1:ps:end));
                    set(s,'FaceColor','flat','LineStyle','none');
                    grid on;
                    plot3(c.X(c.YS==1,1),c.X(c.YS==1,2),c.YS(c.YS==1),'k.','MarkerSize',20);
                    plot3(c.X(c.YS==-1,1),c.X(c.YS==-1,2),c.YS(c.YS==-1)*0,'r.','MarkerSize',20);
                    plot3(c.X(end,1),c.X(end,2),1,'g.','MarkerSize',30);
                    grid on;
                    if (exist('c.YS_gt'))
                        tmp = reshape(c.YS_gt,size(t1));
                        s=surface(t1(1:ps*5:end,1:ps*5:end),t2(1:ps*5:end,1:ps*5:end),tmp(1:ps*5:end,1:ps*5:end));
                        set(s,'FaceColor','none');
                    end
                    axis tight;
                    drawnow;
                    
                    figure(3);clf;hold on;
                    title('Acquisition function');
                    tmp = reshape(c.a,size(c.t1));
                    plot3(c.t(:,1),c.t(:,2),c.PI,'b.')
                    plot3(c.t(:,1),c.t(:,2),c.BU,'r.')
                    plot3(c.X(end,1),c.X(end,2),c.fx,'g.','MarkerSize',40)
                    view([-73,16]);
                    zlim([1e-3,max([c.a;c.fx;1])])
                    grid on;
                    drawnow;
                    if (c.verbose>1)
                        keyboard;
                    end
                elseif (c.n==3)
                    %% 3d plotting
                    figure(1);clf;hold on;
                    
                    isoValues = sort(c.Y(c.YS == 1));
                    num = numel(isoValues);
                    
                    %define the colormap
                    clr = jet(num);
                    colormap(clr)
                    caxis([0 num])
                    colorbar('YTick',(1:num)-0.5, 'YTickLabel',num2str(sort(isoValues(:))))
                    
                    [x,y,z,v] = reducevolume(c.t1,c.t2,c.t3,reshape(c.yc,size(c.t1,1),size(c.t2,1),size(c.t3,1)),[4,4,4]);
                    pC = patch(isosurface(x,y,z,v,0));
                    set(pC, 'FaceAlpha','0.2','EdgeColor',[0,0,0]/255,'EdgeAlpha',0.1);
                    [~,sortInd] = sort(c.Y(c.YS==1));
                    XS = c.X(c.YS==1,:);
                    XS = XS(sortInd,:);
                    scatter3(XS(:,1),XS(:,2),XS(:,3),100,clr,'filled');
                    plot3(c.X(c.YS==-1,1),c.X(c.YS==-1,2),c.X(c.YS==-1,3),'rx','MarkerSize',10,'LineWidth',2);
                    plot3(c.X(end,1),c.X(end,2),c.X(end,3),'go','MarkerSize',15,'LineWidth',5);
                    plot3(c.X(end,1),c.X(end,2),c.X(end,3),'ko','MarkerSize',20,'LineWidth',5);
                    
                    plot3(c.statBCx(end,1),c.statBCx(end,2),c.statBCx(end,3),'ko','MarkerSize',15,'LineWidth',5);
                    plot3(c.statBCx(end,1),c.statBCx(end,2),c.statBCx(end,3),'ro','MarkerSize',20,'LineWidth',5);
                    
                    xlabel('\theta_1');
                    ylabel('\theta_2');
                    zlabel('\theta_3');
                    title(['last candidate: ', mat2str(c.X(end,:),3)])
                    grid on;
                    view([-73,16]);
                    axis tight; axis equal;
                    drawnow;
                    if (c.verbose>1)
                        keyboard;
                    end
                end
                figure(4);clf;hold on;
                iters = 1:length(c.Y);
                plot(iters,c.statBCy,'b--');
                plot(iters(c.YS==1),c.Y(c.YS==1),'gs');
                plot(iters(c.YS==-1),c.Y(c.YS==-1),'rx');
                xlabel('Iterations');
                ylabel('R(\theta)');
                legend('Location','SouthEast','best candidate','successes','failures');
                axis tight;
            end
        end
    end
end
