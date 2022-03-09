using System.Diagnostics;

string inputFile = "input2_08.txt";
string outputFile = "out.txt";

int n = 0;
double eps = 0;
double[] B = null;
double[][] A = null;
double[] output = null;
Stopwatch stopwatch = new Stopwatch();

Input(ref n, ref eps, ref B, ref A);
Solve(args);
Console.ReadLine();


void Input(ref int n, ref double eps, ref double[]? B, ref double[][]? A)
{
    string[] input = File
        .ReadAllText(inputFile)
        .Split(new char[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

    for (int i = 0; i < input.Length; i++)
        input[i] = input[i].Replace('.', ',');

    n = int.Parse(input[0]);
    eps = double.Parse(input[1]);
    
    output = new double[n];
    B = new double[n];
    for (int i = 0; i < n; i++)
    {
        B[i] = double.Parse(input[i + 2]);
        output[i] = B[i];
    }

    input = input.Skip(2 + n).ToArray();

    int idx = 0;
    A = new double[n][];

    for (int i = 0; i < n; i++)
    {
        A[i] = new double[n];
        for (int j = 0; j < n; j++)
        {
            A[i][j] = double.Parse(input[idx]);
            idx++;
        }
    }
}

void Solve(string[] args)
{
    stopwatch.Start();
    MPI.Environment.Run(ref args, communicator =>
    {
        int iterCount = 0;
        double[] X = new double[n];
        while (IsContinue(X, output))
        {
            // сохраняем ответ
            for (int k = 0; k < n; k++)
                X[k] = output[k];

            // синхронизация
            communicator.Barrier();

            // var parts = communicator.Scatter(output);

            iterCount++;

            // Size - кол-во запущенных процессов
            // Rank - номер процесса
            int itersPerProcessor = n / communicator.Size;
            int index = itersPerProcessor * communicator.Rank;

            // остатки в последней порции
            if (communicator.Rank == communicator.Size - 1)
                itersPerProcessor += (n % communicator.Size);

            int end = index + itersPerProcessor;

            double[] partial = new double[itersPerProcessor];
            for (int k = 0; k < itersPerProcessor; k++)
                partial[k] = output[index + k];

            int i = 0;
            for (; index < end; index++)
            {
                double part1 = 0;
                double part2 = 0;

                for (int j = 0; j < index; j++)
                    part1 += A[index][j] * X[j];

                for (int j = index + 1; j < n; j++)
                    part2 += A[index][j] * X[j];


                partial[i] = (B[index] - part1 - part2) / A[index][index];
                i++;
            }

            // Gather производит сборку блоков данных, посылаемых всеми процессами группы, в один массив
            output = communicator
                .Allgather<double[]>(partial)
                .SelectMany(x => x) // [][] => []
                .ToArray();
        }

        if (communicator.Rank == 0)
        { 
            stopwatch.Stop();
            var elapsed = $"Прошло {stopwatch.ElapsedMilliseconds} мс, итераций: {iterCount}\n";
            
            Console.WriteLine(elapsed);
            File.WriteAllText(outputFile, elapsed);

            File.AppendAllText(outputFile, string.Join(' ', output.Select<double, string>(res => res.ToString("f4"))));

            var max = n > 5 ? 5 : n;
            for (int i = 0; i < max; i++)
                Console.WriteLine(output[i].ToString("f4"));
            if (max < n)
                Console.WriteLine("...");

            Console.WriteLine($"Полное решение в файле: {outputFile}");
        }
    });
}

bool IsContinue(double[] X, double[] output)
{
    for (int i = 0; i < n; i++)
        if (eps < Math.Abs(X[i] - output[i]))
            return true;

    return false;
}
