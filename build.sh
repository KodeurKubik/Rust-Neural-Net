rm -r ./neuralnet-demo
mkdir ./neuralnet-demo
cd demo

cp -r ./models ../neuralnet-demo/models
cp -r ./static ../neuralnet-demo/static

cargo build --release --features "relative"
cp ../target/release/demo ../neuralnet-demo/neuralnet-demo-mac
# upx --best --lzma ../neuralnet-demo/neuralnet-demo-mac

cargo build --release --target x86_64-pc-windows-gnu --features "relative"
cp ../target/x86_64-pc-windows-gnu/release/demo.exe ../neuralnet-demo/neuralnet-demo.exe
upx --best --lzma ../neuralnet-demo/neuralnet-demo.exe

cd ..
echo "Open a terminal IN THIS FOLDER! And run the executable from the terminal. You can then open http://localhost:3000 and test the model :)" > ./neuralnet-demo/INSTRUCTIONS.txt
rm -f neuralnet.zip
zip -r neuralnet.zip ./neuralnet-demo