using System;
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
    private static readonly string ModelPath = System.IO.Path.Combine( "model", "cpu-int4-rtn-block-32-acc-level-4" );
    
    [ObservableProperty]
    private string _systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";
    
    [ObservableProperty]
    private string _userPrompt = "Describe the image, and return the string 'STOP' at the end.";
    
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

    [RelayCommand(AllowConcurrentExecutions = false, CanExecute = nameof(IsImagePathValid))]
    private async Task GenerateResponse(CancellationToken cancellationToken)
    {
        AiResponse = string.Empty;

        var imgBytes = await System.IO.File.ReadAllBytesAsync(ImagePath, cancellationToken);
        var fullPrompt = $"<|system|>{SystemPrompt}<|end|><|user|><|image_1|>{UserPrompt}<|end|><|assistant|>";

        var staThread = new Thread(() =>
        {
            try
            {
                using var model = new Model(new Config(ModelPath));
                using var processor = new MultiModalProcessor(model);
                using var tokenizerStream = processor.CreateStream();
                
                using var img = Images.Load(imgBytes);
                using var inputTensors = processor.ProcessImages(fullPrompt, img);

                using var generatorParams = new GeneratorParams(model);
                generatorParams.SetSearchOption("max_length", 3072);

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

        //staThread.SetApartmentState(ApartmentState.STA);
        staThread.IsBackground = false;
        staThread.Start();
        await Task.Run( () => staThread.Join() );
    }
}