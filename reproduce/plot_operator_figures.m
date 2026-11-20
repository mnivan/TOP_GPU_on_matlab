function plot_operator_figures(csvPath, outputDir)
% Plot the two operator figures directly from reproduce/time.m output.
repoRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(csvPath)
    csvPath = fullfile(repoRoot,'reproduce','time_results_all_sizes.csv');
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(repoRoot,'reproduce');
end
T = readtable(csvPath);
assert(height(T)==4 && all(string(T.status)=="COMPLETED"), ...
    'All four matrix-free operator benchmarks must be complete.');
assert(all(string(T.gpuMatrixFreeStatus)=="COMPLETED"));
if ~exist(outputDir,'dir'), mkdir(outputDir); end

sizes = T.numElements;
blue = [0.00 0.20 0.60]; orange = [0.85 0.33 0.10];
gray = [0.20 0.20 0.20]; purple = [0.50 0.20 0.60];

fig = figure('Color','w','Visible','off','Units','centimeters', ...
    'Position',[5 5 14 8]);
ax = axes(fig); hold(ax,'on');
plot(ax,sizes,T.gpuMatrixFreeTotal,'-o','Color',orange,'LineWidth',1.8, ...
    'MarkerSize',6,'MarkerFaceColor','w','DisplayName','Ours');
plot(ax,sizes,T.cpuMatrixFreeTotal,'-s','Color',blue,'LineWidth',1.8, ...
    'MarkerSize',6,'MarkerFaceColor','w','DisplayName','Wang (2025)');
plot(ax,sizes,T.cpuSpmvMean,'-^','Color',gray,'LineWidth',1.8, ...
    'MarkerSize',6,'MarkerFaceColor','w','DisplayName','MATLAB (CPU)');
plot(ax,sizes,T.gpuSpmvMean,'-d','Color',purple,'LineWidth',1.8, ...
    'MarkerSize',6,'MarkerFaceColor','w','DisplayName','MATLAB (GPU)');
set(ax,'XScale','log','YScale','log','FontName','Times New Roman', ...
    'FontSize',11,'LineWidth',1,'TickDir','out','Box','off');
xlabel(ax,'Number of elements'); ylabel(ax,'Time (s)');
lgd = legend(ax,'Location','northwest'); lgd.Box = 'off';
lgd.FontName = 'Times New Roman'; lgd.FontSize = 11;
exportgraphics(fig,fullfile(outputDir,'spmv2.pdf'),'ContentType','vector');
close(fig);

% Step 1 is gathering, Steps 2-3 are element algebra, Step 4 is scattering.
cpu = 100 * [T.cpuGatherMean,T.cpuMatmulMean,T.cpuScatterMean] ./ T.cpuMatrixFreeTotal;
gpu = 100 * [T.gpuGatherMean,T.gpuMatmulMean,T.gpuScatterMean] ./ T.gpuMatrixFreeTotal;
fig = figure('Color','w','Visible','off','Units','centimeters', ...
    'Position',[5 5 14 8]);
ax = axes(fig); hold(ax,'on');
x = (0:3)*1.35; width = 0.34; gap = 0.22;
cpuX = x-(width/2+gap/2); gpuX = x+(width/2+gap/2);
h1 = bar(ax,cpuX,cpu,width,'stacked','EdgeColor',[0.25 0.25 0.25], ...
    'LineWidth',0.6);
h2 = bar(ax,gpuX,gpu,width,'stacked','EdgeColor','none');
barColors = [78 121 167; 242 142 43; 89 161 79]/255;
for k=1:3
    h1(k).FaceColor=barColors(k,:);
    h2(k).FaceColor=barColors(k,:);
end
set(ax,'FontName','Times New Roman','FontSize',11,'LineWidth',1, ...
    'TickDir','out','Box','off','XTick',[],'YTick',0:25:100);
ylim(ax,[0 100]); xlim(ax,[cpuX(1)-width/2-0.15,gpuX(end)+width/2+0.15]);
ylabel(ax,'Time breakdown (%)');
labels = {'0.028M','0.221M','1.77M','14.2M'};
for k=1:4
    text(ax,cpuX(k),-4,'Wang (2025)','HorizontalAlignment','center', ...
        'VerticalAlignment','top','FontName','Times New Roman','FontSize',8, ...
        'Clipping','off');
    text(ax,gpuX(k),-4,'Ours','HorizontalAlignment','center', ...
        'VerticalAlignment','top','FontName','Times New Roman','FontSize',8, ...
        'Clipping','off');
    text(ax,x(k),-13,labels{k},'HorizontalAlignment','center', ...
        'VerticalAlignment','top','FontName','Times New Roman','FontSize',8, ...
        'Clipping','off');
end
lgd = legend(ax,h1,{'Step 1','Steps 2&3','Step 4'},'Location','northoutside');
lgd.Box = 'off'; lgd.Orientation='horizontal'; lgd.FontName='Times New Roman';
exportgraphics(fig,fullfile(outputDir,'break_down2.pdf'),'ContentType','vector');
close(fig);
fprintf('Saved spmv2.pdf and break_down2.pdf to %s\n',outputDir);
end
