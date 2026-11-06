function run_flood_audit(root)
% Execute the unmodified authors' solver; no source positions provided.
addpath(genpath(fullfile(root,'author','utils','omt')));
names={'synthetic','L20_R200_LOS','L20_R200_MP'};
for k=1:numel(names)
    name=names{k};
    input=load(fullfile(root,[name '.mat']));
    begin=tic;
    [M,cluster_vec,trash_mass]=omt_clustering_w_entropy_reg_penalty_logstable_w_trashstate(...
        input.C,input.trash,input.epsilon,input.eta,input.R);
    seconds=toc(begin);
    paper_q=max(M,[],1)';
    mass_residual=max(abs(sum(M,2)+trash_mass-1));
    capacity_residual=max(max(sum(M,1)-input.R*(input.R-1)/2),0);
    objective=sum(input.C.*M,'all')+input.trash*sum(trash_mass)+input.eta*sum(paper_q);
    save(fullfile(root,[name '_author.mat']),'M','cluster_vec','trash_mass','paper_q',...
        'seconds','mass_residual','capacity_residual','objective','-v7');
    fprintf('%s seconds %.3f row residual %.3g capacity %.3g objective %.9g\n',...
        name,seconds,mass_residual,capacity_residual,objective);
end
end
