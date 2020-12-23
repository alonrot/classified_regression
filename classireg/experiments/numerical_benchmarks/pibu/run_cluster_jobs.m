
clear;

addpath('/usr/local/share/htcondor_matlab/')
pause(0.1)

htjob = htcondorjob('PIBUclus');

htjob.mem = 2048;
htjob.bid = 200;

Njobs = 100;
pause_s = 1.0;
fprintf('Adding %d jobs, and waiting %d seconds inbetween each addition ...\n',Njobs,pause_s);
for ii = 1:Njobs
	htjob.addJob(ii);
	pause(pause_s);
end

fprintf('We pause again for 5 seconds before calling htjob.run ...\n')
pause(5);

fprintf('Running %d jobs ...\n',Njobs);
htjob.run;
fprintf('Done!');

which_obj = 'eggs2D'
% which_obj = 'hart6D'
% which_obj = 'micha10D'

fprintf('Saving %d jobs ...\n',Njobs);
for ii = 1:Njobs

	path2save = strcat('./results/',which_obj,'/PIBUclus_job',string(ii),'.mat');
	out = htjob.collectResults(ii);
	fprintf('Saving job %d / %d in %s ... \n',ii,Njobs,path2save);
	save(path2save,'out');
	pause(0.01);

end


htjob.cleanUp