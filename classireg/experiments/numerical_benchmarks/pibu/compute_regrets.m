clear;

addpath('mat2np');

% which_obj = 'micha10D';
% which_obj = 'hart6D';
which_obj = 'eggs2D';

job_name = '/PIBUclus_job';
if strcmp(which_obj,'micha10D')
	yOpt = +9.6601517;
	job_name = '/htcondor_return.';
elseif strcmp(which_obj,'hart6D')
	yOpt = +3.32236801141551 * 10;
elseif strcmp(which_obj,'eggs2D')
	yOpt = 98.;
end

path2open = strcat('./results/',which_obj);

Njobs = 100;
regret_vec = zeros(Njobs,1);
Nfails_vec = zeros(Njobs,1);
Nevals_vec = zeros(Njobs,1);
for ii = 1:Njobs

	if strcmp(which_obj,'micha10D')
		job = load(strcat(path2open,job_name,string(ii),'.mat'));
		job = job.ret{1};
	else
		job = load(strcat(path2open,job_name,string(ii),'.mat'));
		job = job.out;
	end

	% keyboard;

	regret_vec(ii) = yOpt - job.statBCy(end);
	Nfails_vec(ii) = job.statNOF(end);
	Nevals_vec(ii) = length(job.Y);

	% keyboard;

end

% keyboard;
std_noise = 0.01;
assert(all(regret_vec > 0.0 - 3*std_noise));
assert(all(Nevals_vec == Nevals_vec(1)));


Nevals = Nevals_vec(1);
regret_mean = mean(regret_vec);
regret_std = std(regret_vec);
Nfails_mean = mean(Nfails_vec);
Nfails_std = std(Nfails_vec);

fprintf('Nevals: %d \n',Nevals)
fprintf('regret_mean: %d \n',regret_mean)
fprintf('regret_std: %d \n',regret_std)
fprintf('Nfails_mean: %d \n',Nfails_mean)
fprintf('Nfails_std: %d \n',Nfails_std)

%% Save all data in a vector
out2save = [regret_mean,regret_std,Nfails_mean,Nfails_std,Nevals];
path2save = strcat(path2open,'/regret_data.pkl');
mat2np(out2save, path2save, 'float64');
