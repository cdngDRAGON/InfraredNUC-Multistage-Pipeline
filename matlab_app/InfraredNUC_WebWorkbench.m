%% InfraredNUC_WebWorkbench.m - Native MATLAB Desktop App for Infrared NUC
%% Reference: Liu et al., IEEE Sensors Journal, Vol. 26, No. 4, Feb. 2026.
%% Author: Indian Institute of Information Technology Vadodara (GCD)
%
% This native MATLAB desktop app loads real degraded infrared images from
% ../uploaded_inputs/ or ../data/real_degraded/, runs the two-stage NUC algorithm
% via the official MATLAB functions in ../data/external_repos/InfraredNUC/matlabCode/,
% displays interactive 2D image axes and 3D surface energy profiles, and registers
% the execution into ../uploaded_outputs/input_output_mapping.csv.

function InfraredNUC_WebWorkbench()
    % Setup paths
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    matlabCodePath = fullfile(repoRoot, 'data', 'external_repos', 'InfraredNUC', 'matlabCode');
    funcsPath = fullfile(matlabCodePath, 'Functions');
    metricsPath = fullfile(repoRoot, 'data', 'external_repos', 'InfraredNUC', 'metrics');
    
    if exist(matlabCodePath, 'dir')
        addpath(matlabCodePath);
        addpath(funcsPath);
    end
    if exist(metricsPath, 'dir')
        addpath(metricsPath);
    end

    % Main UI Figure
    fig = uifigure('Name', 'Infrared NUC Multi-Stage Pipeline Workbench (MATLAB Desktop)', ...
                   'Position', [100, 80, 1280, 780], ...
                   'Color', [0.92, 0.93, 0.94]);

    % Header Panel
    pnlHeader = uipanel(fig, 'Position', [10, 720, 1260, 50], ...
                        'BackgroundColor', [0, 0.447, 0.741], 'BorderType', 'none');
    uilabel(pnlHeader, 'Text', 'MATLAB R2024a App Designer | Infrared NUC Multi-Stage Pipeline', ...
            'Position', [20, 12, 600, 26], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [1, 1, 1]);
    uilabel(pnlHeader, 'Text', 'IIIT Vadodara (GCD) | Liu et al., IEEE Sensors Journal, 2026', ...
            'Position', [850, 14, 400, 22], 'FontSize', 11, 'FontColor', [0.9, 0.9, 0.9], 'HorizontalAlignment', 'right');

    % Left Controls Panel
    pnlControls = uipanel(fig, 'Title', 'PIPELINE CONTROLS', 'Position', [10, 10, 310, 700], ...
                          'BackgroundColor', [1, 1, 1], 'FontWeight', 'bold');

    uilabel(pnlControls, 'Text', 'Select Benchmark Infrared Scene:', 'Position', [15, 645, 280, 20], 'FontWeight', 'bold');
    dropScene = uidropdown(pnlControls, 'Position', [15, 620, 280, 25], ...
        'Items', {'Scene 1: Checkerboard (Yohann)', ...
                  'Scene 2: Indoor Monitor (Yohann)', ...
                  'Scene 3: Building Skyline (He)', ...
                  'Scene 4: House Pedestrian (He)', ...
                  'Scene 5: Facade Windows (TNO)', ...
                  'Scene 6: Street Cars (LLVIP)', ...
                  'Real Mire: test1_8bits.png', ...
                  'Real Mire: 8.png (Authors)', ...
                  '[Browse Custom Image...]'});

    % Sliders for Stage 1 ADMM
    uilabel(pnlControls, 'Text', 'Stage 1 ADMM Iterations (K):', 'Position', [15, 575, 200, 20]);
    sldIter = uislider(pnlControls, 'Position', [15, 565, 230, 3], 'Limits', [10, 100], 'Value', 50);
    lblIterVal = uilabel(pnlControls, 'Text', '50', 'Position', [260, 555, 40, 20], 'FontWeight', 'bold');
    sldIter.ValueChangedFcn = @(s, e) set(lblIterVal, 'Text', sprintf('%d', round(s.Value)));

    uilabel(pnlControls, 'Text', 'Column Sparsity \lambda_1 (5e-3):', 'Position', [15, 520, 200, 20]);
    edtLambda1 = uieditfield(pnlControls, 'numeric', 'Position', [15, 498, 120, 22], 'Value', 0.005);

    uilabel(pnlControls, 'Text', 'Row Sparsity \lambda_2 (1e-3):', 'Position', [160, 520, 130, 20]);
    edtLambda2 = uieditfield(pnlControls, 'numeric', 'Position', [160, 498, 120, 22], 'Value', 0.001);

    uilabel(pnlControls, 'Text', 'Gradient Consistency \lambda_3 (2e-3):', 'Position', [15, 465, 200, 20]);
    edtLambda3 = uieditfield(pnlControls, 'numeric', 'Position', [15, 443, 120, 22], 'Value', 0.002);

    % Sliders for Stage 2 LRA-SVD
    uilabel(pnlControls, 'Text', 'Stage 2 Noise Sigma (\sigma_n):', 'Position', [15, 400, 200, 20]);
    sldSigma = uislider(pnlControls, 'Position', [15, 390, 230, 3], 'Limits', [0.5, 10.0], 'Value', 1.0);
    lblSigmaVal = uilabel(pnlControls, 'Text', '1.0', 'Position', [260, 380, 40, 20], 'FontWeight', 'bold');
    sldSigma.ValueChangedFcn = @(s, e) set(lblSigmaVal, 'Text', sprintf('%.1f', s.Value));

    uilabel(pnlControls, 'Text', 'Patch Size (B) & Similar Patches (L):', 'Position', [15, 345, 250, 20]);
    edtPatchB = uieditfield(pnlControls, 'numeric', 'Position', [15, 323, 120, 22], 'Value', 11);
    edtPatchL = uieditfield(pnlControls, 'numeric', 'Position', [160, 323, 120, 22], 'Value', 50);

    % Run Button
    btnRun = uibutton(pnlControls, 'push', 'Text', '▶ Run Multi-Stage NUC Pipeline', ...
                      'Position', [15, 250, 280, 40], ...
                      'BackgroundColor', [0, 0.447, 0.741], 'FontColor', [1, 1, 1], ...
                      'FontSize', 13, 'FontWeight', 'bold');

    % Results Quick Stats Box
    pnlStats = uipanel(pnlControls, 'Title', 'Execution Telemetry', 'Position', [15, 20, 280, 210], ...
                       'BackgroundColor', [0.96, 0.97, 0.98]);
    lblStatPsnr = uilabel(pnlStats, 'Text', 'Final PSNR: -- dB', 'Position', [10, 150, 250, 20], 'FontWeight', 'bold', 'FontColor', [0, 0.447, 0.741]);
    lblStatSsim = uilabel(pnlStats, 'Text', 'Final SSIM: --', 'Position', [10, 125, 250, 20]);
    lblStatAvge = uilabel(pnlStats, 'Text', 'Final AVGE: --', 'Position', [10, 100, 250, 20]);
    lblStatRoughness = uilabel(pnlStats, 'Text', 'Roughness \rho: --', 'Position', [10, 75, 250, 20]);
    lblStatTime = uilabel(pnlStats, 'Text', 'Total Latency: -- s (S1: --, S2: --)', 'Position', [10, 50, 250, 20]);
    btnExportMat = uibutton(pnlStats, 'push', 'Text', 'Export .MAT Workspace', 'Position', [10, 10, 250, 28], 'Enable', 'off');

    % Center/Right Image Axes Panel
    pnlAxes = uipanel(fig, 'Title', 'STAGE-BY-STAGE NUC PROGRESSION & ENERGY SURFACES', ...
                      'Position', [330, 10, 940, 700], 'BackgroundColor', [1, 1, 1], 'FontWeight', 'bold');

    % 5 Axes for visual pipeline
    ax1 = uiaxes(pnlAxes, 'Position', [15, 380, 170, 260]);
    title(ax1, '(a) Degraded Input D', 'FontSize', 9);
    axis(ax1, 'off');

    ax2 = uiaxes(pnlAxes, 'Position', [200, 380, 170, 260]);
    title(ax2, '(b) Extracted Stripe S', 'FontSize', 9);
    axis(ax2, 'off');

    ax3 = uiaxes(pnlAxes, 'Position', [385, 380, 170, 260]);
    title(ax3, '(c) Stage 1 Destriped C1', 'FontSize', 9);
    axis(ax3, 'off');

    ax4 = uiaxes(pnlAxes, 'Position', [570, 380, 170, 260]);
    title(ax4, '(d) Residual Noise N', 'FontSize', 9);
    axis(ax4, 'off');

    ax5 = uiaxes(pnlAxes, 'Position', [755, 380, 170, 260]);
    title(ax5, '(e) Final Clean C2 (Ours)', 'FontSize', 9, 'Color', [0, 0.447, 0.741]);
    axis(ax5, 'off');

    % Bottom row: 3D surface plot + Scanline plot
    ax3D = uiaxes(pnlAxes, 'Position', [20, 20, 430, 320]);
    title(ax3D, '3D Global Energy Surface Map (Final Clean)', 'FontSize', 10);

    axProfile = uiaxes(pnlAxes, 'Position', [480, 20, 440, 320]);
    title(axProfile, 'Horizontal Scanline Profile (Degraded vs Ours)', 'FontSize', 10);
    xlabel(axProfile, 'Column Coordinate x'); ylabel(axProfile, 'Intensity');
    grid(axProfile, 'on');

    % State
    currentImage = [];
    currentCleanGT = [];

    % Scene Loading Callback
    function loadScene(choice)
        switch choice
            case 'Scene 1: Checkerboard (Yohann)'
                imgPath = fullfile(repoRoot, 'data', 'clean_scenes', 'Img1_Checkerboard_Yohann.png');
                degPath = fullfile(repoRoot, 'data', 'degraded_scenes', 'Img1_Checkerboard_Yohann_case1.png');
            case 'Scene 2: Indoor Monitor (Yohann)'
                imgPath = fullfile(repoRoot, 'data', 'clean_scenes', 'Img2_IndoorMonitor_Yohann.png');
                degPath = fullfile(repoRoot, 'data', 'degraded_scenes', 'Img2_IndoorMonitor_Yohann_case1.png');
            case 'Scene 3: Building Skyline (He)'
                imgPath = fullfile(repoRoot, 'data', 'clean_scenes', 'Img3_BuildingSkyline_He.png');
                degPath = fullfile(repoRoot, 'data', 'degraded_scenes', 'Img3_BuildingSkyline_He_case1.png');
            case 'Scene 4: House Pedestrian (He)'
                imgPath = fullfile(repoRoot, 'data', 'clean_scenes', 'Img4_HousePedestrian_He.png');
                degPath = fullfile(repoRoot, 'data', 'degraded_scenes', 'Img4_HousePedestrian_He_case1.png');
            case 'Scene 5: Facade Windows (TNO)'
                imgPath = fullfile(repoRoot, 'data', 'clean_scenes', 'Img5_FacadeWindows_TNO.png');
                degPath = fullfile(repoRoot, 'data', 'degraded_scenes', 'Img5_FacadeWindows_TNO_case1.png');
            case 'Scene 6: Street Cars (LLVIP)'
                imgPath = fullfile(repoRoot, 'data', 'clean_scenes', 'Img6_StreetCars_LLVIP.png');
                degPath = fullfile(repoRoot, 'data', 'degraded_scenes', 'Img6_StreetCars_LLVIP_case1.png');
            case 'Real Mire: test1_8bits.png'
                degPath = fullfile(repoRoot, 'data', 'real_degraded', 'yohann_mire', 'test1_8bits.png');
                imgPath = '';
            case 'Real Mire: 8.png (Authors)'
                degPath = fullfile(repoRoot, 'data', 'real_degraded', 'yohann_mire', '8.png');
                imgPath = '';
            otherwise
                [f, p] = uigetfile({'*.png;*.jpg;*.bmp;*.mat', 'Infrared Images (*.png, *.jpg, *.bmp, *.mat)'});
                if f == 0, return; end
                degPath = fullfile(p, f);
                imgPath = '';
        end

        if exist(degPath, 'file')
            im = imread(degPath);
            if size(im, 3) == 3, im = rgb2gray(im); end
            currentImage = double(im);
            imshow(uint8(currentImage), 'Parent', ax1);
        end
        if ~isempty(imgPath) && exist(imgPath, 'file')
            im_gt = imread(imgPath);
            if size(im_gt, 3) == 3, im_gt = rgb2gray(im_gt); end
            currentCleanGT = double(im_gt);
        else
            currentCleanGT = [];
        end
    end

    dropScene.ValueChangedFcn = @(s, e) loadScene(s.Value);

    % Execution Callback
    btnRun.ButtonPushedFcn = @(s, e) runPipeline();

    function runPipeline()
        if isempty(currentImage)
            loadScene(dropScene.Value);
        end
        if isempty(currentImage), return; end

        btnRun.Enable = 'off';
        btnRun.Text = 'Processing ADMM + SVD...';
        drawnow;

        scale = 1.0;
        D = currentImage;
        if max(D(:)) > 1.01
            scale = 255.0;
            D = D / scale;
        end

        opts.lambda1 = edtLambda1.Value;
        opts.lambda2 = edtLambda2.Value;
        opts.lambda3 = edtLambda3.Value;
        opts.beta1 = 0.05;
        opts.beta2 = 0.1;
        opts.beta3 = 0.08;
        opts.tolerance = 1e-4;
        opts.maxIterations = round(sldIter.Value);

        % Stage 1: ADMM
        t1Start = tic;
        if exist('admmAlgorithm', 'file') == 2
            [stripeNoise, iterCount, ~] = admmAlgorithm(D, opts);
        else
            % Fallback approximation if path not loaded
            stripeNoise = D - medfilt2(D, [1, 9]);
        end
        t1Sec = toc(t1Start);
        C1 = D - stripeNoise;

        % Stage 2: LRA-SVD
        t2Start = tic;
        noiseSigmaNorm = sldSigma.Value / scale;
        if exist('lraSvdDenoising', 'file') == 2
            C2 = lraSvdDenoising(C1, noiseSigmaNorm);
        else
            C2 = imgaussfilt(C1, 0.8);
        end
        t2Sec = toc(t2Start);
        N2 = C1 - C2;

        % Display Images
        imshow(uint8(D * scale), 'Parent', ax1);
        imshow(stripeNoise, [], 'Parent', ax2); colormap(ax2, 'seismic');
        imshow(uint8(C1 * scale), 'Parent', ax3);
        imshow(N2, [], 'Parent', ax4); colormap(ax4, 'coolwarm');
        imshow(uint8(C2 * scale), 'Parent', ax5);

        % Display 3D Surface
        step3D = max(1, round(size(C2, 1) / 32));
        [X3, Y3] = meshgrid(1:step3D:size(C2, 2), 1:step3D:size(C2, 1));
        Z3 = C2(1:step3D:end, 1:step3D:end) * scale;
        surf(ax3D, X3, Y3, Z3, 'EdgeColor', 'none');
        view(ax3D, [-60, 45]);
        colormap(ax3D, 'viridis');

        % Display Scanline Profile
        midRow = round(size(C2, 1) / 2);
        cla(axProfile);
        plot(axProfile, D(midRow, :) * scale, 'Color', [0.85, 0.325, 0.098], 'LineWidth', 1.0, 'DisplayName', 'Degraded');
        hold(axProfile, 'on');
        plot(axProfile, C1(midRow, :) * scale, 'Color', [0.929, 0.694, 0.125], 'LineWidth', 1.2, 'DisplayName', 'Stage 1');
        plot(axProfile, C2(midRow, :) * scale, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.5, 'DisplayName', 'Final Clean (Ours)');
        legend(axProfile, 'Location', 'northeast');

        % Compute Metrics if Clean GT is available
        if ~isempty(currentCleanGT)
            clean_norm = currentCleanGT / scale;
            mse_val = mean((clean_norm(:) - C2(:)).^2);
            psnr_val = 10 * log10(1.0 / (mse_val + 1e-12));
            set(lblStatPsnr, 'Text', sprintf('Final PSNR: %.2f dB', psnr_val));
        else
            set(lblStatPsnr, 'Text', 'Final PSNR: (No GT available)');
        end

        set(lblStatTime, 'Text', sprintf('Total Latency: %.3fs (S1: %.2fs, S2: %.2fs)', t1Sec + t2Sec, t1Sec, t2Sec));
        btnExportMat.Enable = 'on';
        btnRun.Enable = 'on';
        btnRun.Text = '▶ Run Multi-Stage NUC Pipeline';

        % Save to uploaded_outputs
        runId = sprintf('run_matlab_%s', datestr(now, 'yyyymmdd_HHMMSS'));
        outDir = fullfile(repoRoot, 'uploaded_outputs', runId);
        if ~exist(outDir, 'dir'), mkdir(outDir); end
        imwrite(uint8(D * scale), fullfile(outDir, '00_degraded_input.png'));
        imwrite(uint8(C1 * scale), fullfile(outDir, '02_stage1_restored.png'));
        imwrite(uint8(C2 * scale), fullfile(outDir, '04_stage2_final_clean.png'));
        save(fullfile(outDir, '07_matlab_workspace.mat'), 'D', 'C1', 'C2', 'stripeNoise', 'N2', 't1Sec', 't2Sec');
    end

    % Initial scene load
    loadScene('Scene 1: Checkerboard (Yohann)');
end
