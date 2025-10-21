using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace SimpleImageAiChat;

public partial class MainViewModel : ObservableObject
{
    private static readonly string ModelPath = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ShyRed", "AI", "models", "Phi-4-multimodal-instruct-onnx", "gpu-int4-rtn-block-32");

    [ObservableProperty]
    private string _downloadText = $"Download the model files and place them in {ModelPath}";
    
    [ObservableProperty]
    private string _systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";
    
    [ObservableProperty]
    private string _userPrompt = "Describe the image.";
    
    [ObservableProperty]
    private string _imagePath = "";
    
    [ObservableProperty]
    private string _aiResponse = "";
    
    [ObservableProperty]
    private IImage? _currentImage = null;
    
    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(GenerateResponseCommand))]
    private bool _isImagePathValid = false;

    partial void OnImagePathChanged(string value)
    {
        var normalized = value.Trim().Trim('\"', '\'', '“', '”', '‘', '’');
        if (!string.Equals(normalized, value, StringComparison.Ordinal))
        {
            SetProperty(ref _imagePath, normalized, nameof(ImagePath));
            value = normalized;
        }
        
        IsImagePathValid = System.IO.File.Exists(value); 
        if (!IsImagePathValid)
            return;

        try
        {
            CurrentImage = new Bitmap(value);
        }
        catch (Exception e)
        {
            AiResponse = e.Message;
        }
    }

    private async Task DownloadModelFiles(CancellationToken cancellationToken)
    {
        try
        {
            Directory.CreateDirectory(ModelPath);

            // Read embedded resource (model_download_urls.txt)
            var assembly = Assembly.GetExecutingAssembly();
            // Adjust resource name if your default namespace/folder differs
            var resourceName = FindResourceName(assembly, "model_download_urls.txt");
            if (resourceName is null)
            {
                AiResponse = "Could not find embedded resource: model_download_urls.txt";
                return;
            }

            string[] urls;
            using (var stream = assembly.GetManifestResourceStream(resourceName)!)
            using (var reader = new StreamReader(stream))
            {
                var content = await reader.ReadToEndAsync();
                urls = content.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            }

            if (urls.Length == 0)
            {
                AiResponse = "No URLs found in model_download_urls.txt";
                return;
            }

            long totalBytesToDownload = 0;
            var http = new HttpClient();
            // Probe sizes (HEAD) to show a total; if it fails, continue anyway
            foreach (var url in urls)
            {
                try
                {
                    using var head = new HttpRequestMessage(HttpMethod.Head, url);
                    using var resp = await http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                    if (resp.IsSuccessStatusCode && resp.Content.Headers.ContentLength.HasValue)
                        totalBytesToDownload += resp.Content.Headers.ContentLength.Value;
                }
                catch
                {
                    // ignore size probe errors
                }
            }

            long totalDownloaded = 0;
            int fileIndex = 0;

            foreach (var url in urls)
            {
                cancellationToken.ThrowIfCancellationRequested();
                fileIndex++;

                var targetFile = Path.Combine(ModelPath, Path.GetFileName(new Uri(url).LocalPath));
                // Skip if already present and non-empty
                if (File.Exists(targetFile) && new FileInfo(targetFile).Length > 0)
                {
                    AiResponse = $"[{fileIndex}/{urls.Length}] Skipped {Path.GetFileName(targetFile)} (already downloaded)";
                    continue;
                }

                AiResponse = $"[{fileIndex}/{urls.Length}] Downloading {Path.GetFileName(targetFile)}...";
                using var resp = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                resp.EnsureSuccessStatusCode();

                var contentLength = resp.Content.Headers.ContentLength ?? 0;
                await using var src = await resp.Content.ReadAsStreamAsync(cancellationToken);
                await using var dst = File.Create(targetFile);

                var buffer = new byte[81920];
                int read;
                long fileDownloaded = 0;
                var lastUpdate = 0L;

                while ((read = await src.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
                {
                    await dst.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                    fileDownloaded += read;
                    totalDownloaded += read;

                    // Throttle UI updates to avoid spamming
                    if (fileDownloaded - lastUpdate >= 256 * 1024 || read == 0)
                    {
                        lastUpdate = fileDownloaded;
                        var filePct = contentLength > 0 ? (int)(fileDownloaded * 100 / contentLength) : -1;
                        var totalPct = totalBytesToDownload > 0 ? (int)(totalDownloaded * 100 / totalBytesToDownload) : -1;

                        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                        {
                            AiResponse = totalPct >= 0
                                ? $"[{fileIndex}/{urls.Length}] {Path.GetFileName(targetFile)}: {filePct}% | Total: {totalPct}%"
                                : $"[{fileIndex}/{urls.Length}] {Path.GetFileName(targetFile)}: {fileDownloaded / (1024 * 1024)} MiB";
                        });
                    }
                }

                AiResponse = $"[{fileIndex}/{urls.Length}] Completed {Path.GetFileName(targetFile)}";
            }

            AiResponse += Environment.NewLine + "All model files are ready.";
        }
        catch (OperationCanceledException)
        {
            AiResponse = "Download canceled.";
        }
        catch (Exception ex)
        {
            AiResponse = "Download failed: " + ex.Message;
        }
    }

    private static string? FindResourceName(Assembly assembly, string shortName)
    {
        var names = assembly.GetManifestResourceNames();
        return names.FirstOrDefault(n => n.EndsWith(shortName, StringComparison.OrdinalIgnoreCase));
    }

    [RelayCommand(AllowConcurrentExecutions = false, CanExecute = nameof(IsImagePathValid), IncludeCancelCommand = true)]
    private async Task GenerateResponse(CancellationToken cancellationToken)
    {
        AiResponse = string.Empty;

        if (!File.Exists(Path.Combine(ModelPath, "tokenizer.json")))
        {
            Directory.CreateDirectory(ModelPath);
            AiResponse = "Model files missing. Starting download...";
            await DownloadModelFiles(cancellationToken);
            // If still missing after download, abort
            if (!File.Exists(Path.Combine(ModelPath, "tokenizer.json")))
            {
                AiResponse += Environment.NewLine + "Model files are not available.";
                return;
            }
        }

        var imgBytes = await System.IO.File.ReadAllBytesAsync(ImagePath, cancellationToken);
        var fullPrompt = $"<|system|>{SystemPrompt}<|end|><|user|><|image_1|>{UserPrompt}<|end|><|assistant|>";
        
        var staThread = new Thread(() =>
        {
            try
            {
                var config = new Config(ModelPath);

                using var model = new Model(config);
                using var processor = new MultiModalProcessor(model);
                using var tokenizerStream = processor.CreateStream();
                
                using var img = Images.Load(imgBytes);
                using var inputTensors = processor.ProcessImages(fullPrompt, img);

                using var generatorParams = new GeneratorParams(model);
                generatorParams.SetSearchOption("max_length", 8192);

                using var generator = new Generator(model, generatorParams);
                generator.SetInputs(inputTensors);

                while (!generator.IsDone() && !cancellationToken.IsCancellationRequested)
                {
                    generator.GenerateNextToken();
                    var seq = generator.GetSequence(0)[^1];
                    var token = tokenizerStream.Decode(seq);

                    Avalonia.Threading.Dispatcher.UIThread.Post(() => AiResponse += token);
                }
            }
            catch (Exception ex)
            {
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    AiResponse += Environment.NewLine + Environment.NewLine + ex.Message;
                });
            }
        });

        staThread.IsBackground = false;
        staThread.Start();
        await Task.Run( () => staThread.Join() );
    }
}