import { cp, mkdir } from 'node:fs/promises';
import path from 'node:path';

const sourceDir = path.resolve('static');
const outputDir = path.resolve('public');

await mkdir(outputDir, { recursive: true });
await cp(sourceDir, outputDir, { recursive: true });
console.log(`Copied ${sourceDir} -> ${outputDir}`);